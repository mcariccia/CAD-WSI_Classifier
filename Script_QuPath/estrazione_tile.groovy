import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.common.GeneralTools
import qupath.lib.common.ColorTools
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

// ==============================================================================
// 1. CONFIGURAZIONE
// ==============================================================================
String pathOutput = "E:/MarcoCariccia/Dataset/Dataset_Tiles"
int outputTileSize = 512       
double downsample = 2.0        // Zoom a 40x, gold standard per cervical CAD

int baseSize = (int)(outputTileSize * downsample) 
int step = (int)(baseSize * 0.75) 

// ==============================================================================
// 2. MAPPATURA CLASSI E COLORI
// ==============================================================================
def classMapping = [
    "Tissue"             : [id: 255, color: ColorTools.BLACK], // Tessuto sconosciuto (BLACK)
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

print "\n--- Elaborazione WSI: ${cleanName} ---"

int blackColor = 0xFF000000.intValue() 
int whiteColor = 0xFFFFFFFF.intValue() 

// ==============================================================================
// 4. RILEVAMENTO TESSUTO (CON DIAGNOSTICA)
// ==============================================================================
try {
    def tissueClass = getPathClass("Tissue")
    def oldTissue = hierarchy.getAnnotationObjects().findAll { it.getPathClass() == tissueClass }
    removeObjects(oldTissue, true)

    // Filtro abbassato a 10.0 per impedire a QuPath di cancellare il tessuto
    createAnnotationsFromPixelClassifier("Rilevatore_Tessuto", 10.0, 10.0)
    
    // --- DIAGNOSTICA ---
    def newTissue = hierarchy.getAnnotationObjects().findAll { it.getPathClass() == tissueClass }
    print " -> DIAGNOSTICA: Generate ${newTissue.size()} aree di 'Tissue' bianco."
    
    if (newTissue.isEmpty()) {
        print " ❌ ERRORE CRITICO: Il Rilevatore non sta creando nulla! Le maschere non avranno il bianco."
    }

    resolveHierarchy() 
} catch (Exception e) {
    print "ATTENZIONE: Classificatore 'Rilevatore_Tessuto' fallito per ${cleanName}."
    return
}

def targetClasses = ["CIN1", "Endocervical glands", "HSIL", "Normal Mucosa", "Stroma"]
def targetAnnotations = hierarchy.getAnnotationObjects().findAll { 
    it.getPathClass() != null && targetClasses.contains(it.getPathClass().getName()) 
}

if (targetAnnotations.isEmpty()) {
    print "Nessuna annotazione patologica trovata. Salto."
    return
}

// ==============================================================================
// 5. SETUP GESTIONE FILE E FORZATURA MASCHERA
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
// 6. ESTRAZIONE TILE MULTI-CORE 
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

requests.parallelStream().forEach { req ->
    int x = req.x
    int y = req.y
    def region = RegionRequest.createInstance(server.getPath(), downsample, x, y, baseSize, baseSize)
    
    BufferedImage maskImg = null
    try { maskImg = labelServer.readBufferedImage(region) } catch (Exception e) { return }
    
    int[] maskPixels = maskImg.getRGB(0, 0, outputTileSize, outputTileSize, null, 0, outputTileSize)
        int pixelUtili = 0
        
        for (int p : maskPixels) {
            if (p != blackColor && p != whiteColor) {
                pixelUtili++
            }
        }
        
        double percentualeTessuto = (double) pixelUtili / maskPixels.length
        
        if (percentualeTessuto < 0.05) return
    
    try {
        BufferedImage rgbImg = server.readBufferedImage(region)
        String filename = "tile_x" + x + "_y" + y + ".png" 
        ImageIO.write(rgbImg, "png", new File(imgDir, filename))
        ImageIO.write(maskImg, "png", new File(maskDir, filename))
        totalTilesSaved.incrementAndGet()
    } catch (Exception e) {}
}

print "    Completato! Estratti " + totalTilesSaved.get() + " tile utili."
System.gc()