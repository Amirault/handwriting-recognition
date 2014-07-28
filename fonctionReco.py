# -*- coding: utf-8 -*-
"""
Auteurs : AMIRAULT TONY <> PHAM QUOC DAT
"""
import numpy as np,Image
import csv
import pickle 
import matplotlib
        
def ModulePretraitementACP(nbImg):
# permet la recuperation des composantes principale
    for i in range(0,nbImg):
        dataImg = Image.open("train/%04d.pgm" %i) #Recuperation de l'image # lecture de l'image d'apprentissage courante
        vecteurImg = np.asarray(dataImg).flatten()#Conversion de l'image en vecteur colonne
        # Concatenation de chaque image (vecteur ligne) dans une unique matrice  
        if (i == 0):
            MatriceVecteursImg = vecteurImg
        else:
            MatriceVecteursImg = np.vstack((MatriceVecteursImg,vecteurImg))    

#   Calcul de la moyenne pour recentrage    
#   Inutile graçe au gaussiènne       
#    nu = np.mean(MatriceVecteursImg.T, axis=1) # calcul de la moyenne
#    output = open('Resource/nu.csv', 'wb')
#    pickle.dump(nu, output)
#    output.close()
    MatriceCov = np.cov(MatriceVecteursImg.T) # calcul la matrice de covariance
    valP,vectP = np.linalg.eigh(MatriceCov) #recupere les valeurs propres et vecteurs propres    
    listValP= valP.tolist() # Valeur propre en liste
    listVectP = vectP.T.tolist()# Vecteur propre en liste
    DicoValVectP= zip(listValP,listVectP ) #association valeurs propres et vecteurs propres
    DicoValVectP.sort(reverse=True) #Tri dans l'ordre decroissant
    vectP =[x[1] for x in DicoValVectP ]#recupere les vecteurs propre trier dans le bon ordre
    
    return np.array(vectP),MatriceVecteursImg

def ModuleApprentissageAvecAcp(nbImg,nbDim,vectP,MatriceData):
# Apprentissage de chaque classe avec simplification acp
    
    vectP = vectP[0:nbDim,:]### reduction du nombre de composante pour la projection
    listApp = recupList(nbImg,"train.txt")
    listApp.sort() # tri selon les listClasseImgs
    listClasseImg =[x[0] for x in listApp ] #recupere les etiquettes dans le bon ordre
    listNomImg = [x[1] for x in listApp ] #recupere les images dans le bon ordre
    indexClass = []
    offset = 0
    Moyenne = []
    MatriceCov = []

    for i in range(10):# on boucle pour chaque classe
        indexClass.append(listClasseImg.count(str(i)))
        for j in range(indexClass[i]):
#            dataImg = Image.open("train/"+listNomImg[offset+j]+".pgm")
#            #Conversion de l'image en vecteur colonne
#            vecteurImg = np.asarray(dataImg).flatten()
            # projection de l'image d'apprentissage courante via ACP
            vecteurImg = np.dot(vectP,MatriceData[(int(listNomImg[offset+j])),:])
            # Concatenation de chaque image dans une unique matrice    
            if (j == 0):
                MatriceVecteursImg = vecteurImg
            else:
                MatriceVecteursImg = np.vstack((MatriceVecteursImg,vecteurImg))
           
        offset = offset + indexClass[i]
        Moyenne.append(MatriceVecteursImg.mean(0))
        MatriceCov.append(np.cov(MatriceVecteursImg.T))
    
    return Moyenne,MatriceCov;
    
def recupList(nbImg,nomFichier):
# Recuperation de la liste des images d'entrainement associées à leurs classes

    mon_fichier = open(nomFichier, "r")
    contenu = mon_fichier.read()
    contenu = contenu.split()
    listNomImg = []
    listClasseImg = []
    for i in range(2*nbImg):
        if (i%2 == 0):
            listNomImg.append(contenu[i])
        else:
            listClasseImg.append(contenu[i])
        
    return zip(listClasseImg,listNomImg) # creation d'une association listClasseImg -> listNomImg
    
def ModuleRecoImg(nbImgTest,nbDim,vectP,Moyenne,MatriceCov):
# Permet d'appliquer la loi normale et de classifier les données

    listTest = recupList(nbImgTest,"test.txt")
    listClasseImg =[x[0] for x in listTest ] #recupere les etiquettes dans le bon ordre
    # contiendrat la probabilité d'une image pour chaque classe
    resultLoiNormal = np.zeros(10) 
    # reduction du nombre de composante pour la projection
    vectP = vectP[0:nbDim,:]
    
    confusion = np.zeros((10,10));
    nbImageMalClasse = np.zeros((10))
    nbImageParClasse = np.zeros((10))
    tauxErreurGlobal = 0.0
    tauxErreurParClasse = np.zeros((10))
    terme1 = []
    invSigma = []
    
    for j in range(10):    
        terme1.append(-(np.log10(np.linalg.det(MatriceCov[j])))/2);
        invSigma.append(np.linalg.inv(MatriceCov[j]));
        
    for i in range(0,nbImgTest):
            imgTest = Image.open("test/%04d.pgm" %i) # lecture de l'image test courante
            imgTest = np.asarray(imgTest).flatten()#Conversion de l'image en vecteur colonne 
            imgTest = np.dot(vectP,imgTest)
            for j in range(10): 
                resultLoiNormal[j] = (np.subtract(terme1[j],(np.dot(((np.subtract(imgTest,Moyenne[j])).T)/2,np.dot(invSigma[j],np.subtract(imgTest,Moyenne[j]))))))     
            numClasse = resultLoiNormal.argmax()# recuperation de la classe avec la plus haute proba
            if (str(numClasse) != listClasseImg[i]):# on compte les images mal classées
                nbImageMalClasse[int(listClasseImg[i])] = nbImageMalClasse[int(listClasseImg[i])] + 1.0;
            nbImageParClasse[int(listClasseImg[i])] = nbImageParClasse[int(listClasseImg[i])] + 1
            # On ajoute les elements à la matrice de confusion
            confusion[listClasseImg[i],numClasse] = confusion[listClasseImg[i],numClasse] +1
    
    for i in range(0,10):
        tauxErreurParClasse[i] = (nbImageMalClasse[i]/nbImageParClasse[i])*100;
        tauxErreurGlobal = tauxErreurGlobal + nbImageMalClasse[i];#Calcul du taux d'erreur global
    tauxErreurGlobal = (tauxErreurGlobal / nbImgTest)*100;
    
    return confusion,tauxErreurGlobal,tauxErreurParClasse