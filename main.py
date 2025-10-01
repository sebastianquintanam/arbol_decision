# ==============================
# 1) Importar librerías
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ==============================
# 2) Cargar dataset
# ==============================
df = pd.read_csv("data/cuentas_ahorro_dataset.csv")

# ==============================
# 3) Separar variables
# ==============================
X = df.drop(columns=["cuenta_recomendada"])
y = df["cuenta_recomendada"]

num_cols = ["horizonte_meses", "saldo_promedio_millones"]
cat_cols = ["prioridad_rentabilidad", "prioridad_liquidez", "tolerancia_costo",
            "requiere_seguro_depositos", "usa_cashback"]

# ==============================
# 4) Preprocesamiento
# ==============================
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ==============================
# 5) Modelo con pipeline
# ==============================
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=42))
])

# ==============================
# 6) División train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ==============================
# 7) Entrenar modelo
# ==============================
clf.fit(X_train, y_train)

# ==============================
# 8) Evaluación
# ==============================
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ==============================
# 9) Visualizaciones
# ==============================

# --- Versión resumida (para exposición en clase) ---
plt.figure(figsize=(18,10))
plot_tree(
    clf.named_steps["model"], 
    filled=True, 
    rounded=True,
    class_names=clf.classes_, 
    feature_names=num_cols + list(clf.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(cat_cols)),
    max_depth=3   # 👈 solo muestra 3 niveles
)
plt.title("Árbol de decisión (resumido para exposición)")
plt.savefig("arbol_resumido.png", dpi=300, bbox_inches="tight")
plt.show()


# --- Versión completa (para informe escrito) ---
plt.figure(figsize=(40,25))
plot_tree(
    clf.named_steps["model"], 
    filled=True, 
    rounded=True,
    class_names=clf.classes_, 
    feature_names=num_cols + list(clf.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(cat_cols))
)
plt.title("Árbol de decisión completo")
plt.savefig("arbol_completo.png", dpi=400, bbox_inches="tight")
plt.show()
