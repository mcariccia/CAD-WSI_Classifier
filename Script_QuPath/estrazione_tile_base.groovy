import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.common.GeneralTools
import qupath.lib.common.ColorTools
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.ForkJoinPool

// ==============================================================================
// 1. CONFIGURAZIONE
// ==============================================================================
String pathOutput = "E:/Tirocinio/Dataset/Dataset_Tiles"
int outputTileSize = 512       
double downsample = 2.0        

int baseSize = (int)(outputTileSize * downsample)   // Calcola 1024
int step = (int)(baseSize * 0.75)                   // Calcola 768 (25% di overlap)                

// ==============================================================================
// 2. MAPPATURA CLASSI E COLORI
// ==============================================================================
def classMapping = [
    "Tissue"             : [id: 255, color: ColorTools.BLACK], // Tessuto generico non patologico
    "CIN1"               : [id: 1, color: ColorTools.makeRGB(255, 192, 203)], 
    "Endocervical glands": [id: 2, color: ColorTools.GREEN],                  
    "HSIL"               : [id: 3, color: ColorTools.BLUE],                   
    "Normal Mucosa"      : [id: 4, color: ColorTools.YELLOW],                 
    "Stroma"             : [id: 5, color: ColorTools.CYAN]                    
]

// ==============================================================================
// 3. INIZIALIZZAZIONE 
// ==============================================================================
def imageData = getCurrentImageData()
if (imageData == null) {
    print "ERRORE: Usa 'Run for project' per avviare l'elaborazione batch!"
    return
}

def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()
String cleanName = GeneralTools.stripExtension(getProjectEntry().getImageName())

print "\n======================================================="
print "--- Elaborazione WSI: ${cleanName} ---"

int blackColor = 0xFF000000.intValue() 
int whiteColor = 0xFFFFFFFF.intValue() 

// ==============================================================================
// 4. CONTROLLO TESSUTO MANUALE (ZERO COLLI DI BOTTIGLIA)
// ==============================================================================
def tissueClass = getPathClass("Tissue")
def existingTissue = hierarchy.getAnnotationObjects().findAll { it.getPathClass() == tissueClass }

if (existingTissue.isEmpty()) {
    print " ❌ DA FARE A MANO: Nessuna annotazione 'Tissue' trovata! Salto questa WSI."
    return // Esce e passa alla prossima WSI per non estrarre tile sbagliati
} else {
    print " ✅ DIAGNOSTICA: Trovate ${existingTissue.size()} annotazioni 'Tissue' (Fatte a mano)."
}

resolveHierarchy() 

// ==============================================================================
// 5. CONTROLLO ANNOTAZIONI PATOLOGICHE
// ==============================================================================
def targetClasses = ["CIN1", "Endocervical glands", "HSIL", "Normal Mucosa", "Stroma"]
def targetAnnotations = hierarchy.getAnnotationObjects().findAll { 
    it.getPathClass() != null && targetClasses.contains(it.getPathClass().getName()) 
}

if (targetAnnotations.isEmpty()) {
    print "Nessuna annotazione patologica (Ground Truth) trovata. Salto alla prossima WSI."
    return
}

// ==============================================================================
// 6. SETUP GESTIONE FILE E MASCHERE
// ==============================================================================
def labelBuilder = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) 
    .downsample(downsample)
    .multichannelOutput(false) 
    
classMapping.each { className, props -> 
    labelBuilder.addLabel(className, props.id, props.color)
}
def labelServer = labelBuilder.build()

File wsiDir = new File(pathOutput, cleanName)
File imgDir = new File(wsiDir, "images")
File maskDir = new File(wsiDir, "masks")

if (!imgDir.exists()) imgDir.mkdirs()
if (!maskDir.exists()) maskDir.mkdirs()

// ==============================================================================
// 7. ESTRAZIONE TILE (MULTI-THREADING CONTROLLATO A 4 CORE)
// ==============================================================================
double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE, maxX = 0, maxY = 0
for (a in targetAnnotations) {
    def roi = a.getROI()
    if (roi.getBoundsX() < minX) minX = roi.getBoundsX()
    if (roi.getBoundsY() < minY) minY = roi.getBoundsY()
    if (roi.getBoundsX() + roi.getBoundsWidth() > maxX) maxX = roi.getBoundsX() + roi.getBoundsWidth()
    if (roi.getBoundsY() + roi.getBoundsHeight() > maxY) maxY = roi.getBoundsY() + roi.getBoundsHeight()
}

def requests = []
for (int y = (int)minY; y < maxY - baseSize; y += step) {
    for (int x = (int)minX; x < maxX - baseSize; x += step) {
        requests << [x: x, y: y]
    }
}

AtomicInteger totalTilesSaved = new AtomicInteger(0)
AtomicInteger totalRequestsProcessed = new AtomicInteger(0)

print " -> Calcolo griglia completato. Inizio scansione di ${requests.size()} potenziali tile in parallelo (4 Core)..."

def customThreadPool = new ForkJoinPool(4)

customThreadPool.submit({
    requests.parallelStream().forEach { req ->
        
        int processed = totalRequestsProcessed.incrementAndGet()
        
        if (processed % 500 == 0) {
            print "    [IN CORSO] Analizzati ${processed}/${requests.size()} tile... (Salvati finora: ${totalTilesSaved.get()})"
        }

        int x = req.x
        int y = req.y
        def region = RegionRequest.createInstance(server.getPath(), downsample, x, y, baseSize, baseSize)
        
        BufferedImage maskImg = null
        try { 
            maskImg = labelServer.readBufferedImage(region) 
        } catch (Exception e) { 
            return 
        }
        
        int w = maskImg.getWidth(), h = maskImg.getHeight()
        int[] maskPixels = maskImg.getRGB(0, 0, w, h, null, 0, w)
        int pixelUtili = 0
        
        for (int p : maskPixels) {
            if (p != blackColor && p != whiteColor) {
                pixelUtili++
            }
        }
        
        double percentualeTessuto = (double) pixelUtili / maskPixels.length
        
        if (percentualeTessuto < 0.0133) return
        
        try {
            BufferedImage rgbImg = server.readBufferedImage(region)
            String filename = "tile_x" + x + "_y" + y + ".png" 
            ImageIO.write(rgbImg, "png", new File(imgDir, filename))
            ImageIO.write(maskImg, "png", new File(maskDir, filename))
            
            totalTilesSaved.incrementAndGet()
            
        } catch (Exception e) {}
    }
}).get()

customThreadPool.shutdown()

int scartati = requests.size() - totalTilesSaved.get()
print "✅ COMPLETATO! WSI: ${cleanName}"
print "   - Tile totali analizzati: ${requests.size()}"
print "   - Tile SALVATI (utili): ${totalTilesSaved.get()}"
print "   - Tile SCARTATI (<1.33% patologia o vuoti): ${scartati}"
print "=======================================================\n"

System.gc()