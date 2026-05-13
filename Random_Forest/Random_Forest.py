import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report

#================================================================================

def under_sampling(df_train_nb, random_state = 42):
    print("\n--- Distribuzione INIZIALE (Train) ---")
    print(df_train_nb['target'].value_counts())

    min_class_count = df_train_nb['target'].value_counts().min()
    print(f"\nClasse minoritaria ha {min_class_count} campioni. Bilancio tutte le classi a questo valore...")

    df_balanced = df_train_nb.groupby('target').sample(n=min_class_count, random_state=42)

    df_train = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("\n--- Distribuzione BILANCIATA (Train) ---")
    print(df_balanced['target'].value_counts())

    return df_train

#=============================================================================000


def under_sampling_kmeans(df_train_nb, samples_per_class=5000, random_state=42):
    print("--- Distribuzione INIZIALE ---")
    print(df_train_nb['target'].value_counts())

    X_out = []
    y_out = []
    
    classes = df_train_nb['target'].unique()

    print(f"\nInizio K-Means Undersampling (Target: {samples_per_class} campioni per classe)...")
    print("NOTA: Questa operazione potrebbe richiedere qualche minuto a seconda della CPU.\n")
    
    for cls in classes:
        df_cls = df_train_nb[df_train_nb['target'] == cls]
        
        feature_cols = df_cls.columns.drop('target')
        X_cls = df_cls[feature_cols].values
        
        print(f"Classe {cls} ({len(X_cls)} pixel)", end="")
        
        if len(X_cls) <= samples_per_class:
            print(f" -> Troppo pochi, li tengo tutti.")
            X_out.append(X_cls)
            y_out.append(np.full(len(X_cls), cls))

        else:
            print(f" -> Calcolo cluster K-Means in corso...")
            
            kmeans = KMeans(
                n_clusters=samples_per_class,
                random_state=random_state,
                n_init=1 
            ).fit(X_cls)
            
            nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_cls)
            _, nearest_idx = nn.kneighbors(kmeans.cluster_centers_)
            
            selected_X = X_cls[nearest_idx[:, 0]]
            selected_y = np.full(samples_per_class, cls)
            
            X_out.append(selected_X)
            y_out.append(selected_y)

    X_final = np.vstack(X_out)
    y_final = np.concatenate(y_out)

    df_balanced = pd.DataFrame(X_final, columns=feature_cols)
    df_balanced['target'] = y_final

    df_train = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("\n--- Distribuzione BILANCIATA (Train) ---")
    print(df_train['target'].value_counts())

    return df_train

#=====================================================================================


if __name__ == "__main__":
    
    df_train_nb = pd.read_csv("Dataset_RF/train_split_RF_dataset.csv")
    df_test = pd.read_csv("Dataset_RF/test_split_RF_dataset.csv")

    #df_train = under_sampling_kmeans(df_train_nb)
    df_train = under_sampling(df_train_nb)

    clf = RandomForestClassifier(
        n_estimators=250,        
        max_depth=20,            
        min_samples_split=5,     
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )

    X_train = df_train.drop('target', axis=1)
    y_train = df_train['target']
    X_test = df_test.drop('target', axis=1)
    y_test = df_test['target']

    clf.fit(X_train, y_train)

    print("====================================")
    print("   VALUTAZIONE")
    print("====================================")

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))