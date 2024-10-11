import tkinter as tk
from tkinter import filedialog, Toplevel, ttk, messagebox
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Fonction de détcetion des faux billets
def detect_faux_billets(df_examen, reg_log1_loaded, seuil=0.5):
    prediction_proba = reg_log1_loaded.predict(df_examen)
    prediction_seuil = (prediction_proba >= seuil).astype(int)
    Resultat = df_examen.copy()
    Resultat['Prédiction'] = prediction_seuil.replace({1: 'Vrai billet', 0: 'Faux billet'})
    Resultat['Fiabilité (%)'] = np.where(prediction_seuil == 1, round(prediction_proba, 2), round(1 - prediction_proba, 2)) * 100 
    resultat_par_type = Resultat.groupby('Prédiction')['diagonal'].count()
    return Resultat, resultat_par_type, prediction_seuil

# Fonctiond d'affichage des resultats de la prédiction
def display_results(resultat, prediction_seuil):
    
    #initialisation de la fenetre 
    result_window = Toplevel(root)
    result_window.title("Résultats de Prédiction")
    result_window.geometry('800x750')

    # Frame pour le résumé et le tableau
    summary_frame = tk.Frame(result_window, bg='white', padx=10, pady=10)
    summary_frame.pack(fill=tk.BOTH, expand=True)

    # Label pour le résumé
    total_vrais = resultat['Prédiction'].value_counts().get('Vrai billet', 0)
    total_faux = resultat['Prédiction'].value_counts().get('Faux billet', 0)
    summary_text = f"Nombre de vrais billets: {total_vrais} | Nombre de faux billets: {total_faux}"
    summary_label = tk.Label(summary_frame, text=summary_text, font=("Arial", 14), bg='white')
    summary_label.pack(pady=10)

    # Affichage du resultat sous forme de dataframe
    tree_frame = tk.Frame(result_window, bg='white')
    tree_frame.pack(fill=tk.BOTH, expand=True)

    tree = ttk.Treeview(tree_frame, columns=['Index'] + list(resultat.columns), show="headings", height=10)
    tree.heading("Index", text="Index")
    for column in list(resultat.columns):
        tree.heading(column, text=column)
        tree.column(column, width=80, anchor='center')

    for index, row in resultat.iterrows():
        values = [index] + list(row)
        tree.insert("", "end", values=values)

    for col in tree['columns']:
        tree.column(col, width=80, anchor='center')

    tree.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    
    # Affichage du pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    counts = resultat['Prédiction'].value_counts()
    labels = [f"{label} ({count})" for label, count in zip(counts.index, counts.values)]
    sizes = counts.values
    ax.pie(sizes, labels=labels, autopct='%.1f%%', startangle=90)
    ax.axis('equal')

    chart_frame = tk.Frame(result_window, bg='white')
    chart_frame.pack(fill=tk.BOTH, expand=True)
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Fonction pour charger les données d'examen et effectuer la prédiction
def load_data_and_predict(reg_log1_model_path='reg_log1_model.joblib'):
    file_path = filedialog.askopenfilename()
    if file_path:
        df_examen = pd.read_csv(file_path, sep=',')
        # Chargement du modèle de régression logistique
        reg_log1_loaded = joblib.load(reg_log1_model_path)
        resultat, resultat_par_type, prediction_seuil = detect_faux_billets(df_examen, reg_log1_loaded)
        messagebox.showinfo(message='Prédiction terminée')
        display_results(resultat, prediction_seuil)

# Interface tkinter
root = tk.Tk()
root.title("Détection de Faux Billets")
root.geometry('300x200')
root.configure(bg='blue')

frame1 = tk.Frame(root)
frame1.pack(pady=10)
frame1.configure(bg='blue')

# Affichage du logo
logo_path = 'C:/Users/Armel/Desktop/Formation_OC/Projet_10/logo.png'  
logo = Image.open(logo_path)
logo = logo.resize((167, 47))  
logo_img = ImageTk.PhotoImage(logo)
logo_label = tk.Label(frame1, image=logo_img)
logo_label.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)
frame.configure(bg='blue')

# Bouton pour charger les données et prédire
predict_button = tk.Button(frame, text="Charger les données et prédire", command=load_data_and_predict)
predict_button.configure(bg='white')
predict_button.pack(pady=10)

# Lancer la boucle principale Tkinter
root.mainloop()
