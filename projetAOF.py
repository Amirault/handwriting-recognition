# -*- coding: utf-8 -*-

"""
Auteurs : AMIRAULT TONY <> PHAM QUOC DAT
"""
import numpy as np,Image
import csv
import pickle
import fonctionReco as prg
import os.path
import matplotlib.pyplot as plt

# Initialisation des variables
nbDim = 0
nbImgApp = 0
nbImgTest = 0
choixMenu = 4
vectP = 0
Moyenne = 0
MatriceCov = 0
MatriceData = 0

# Création du répertoire de sauvegarde s'il nexiste pas
if(os.path.exists("Resource") == False):
    os.mkdir("Resource")
else:
    #lecture des données sauvegardées
    if(os.path.isfile('Resource/nbImgApp.csv') == True):
        pkl_file = open('Resource/nbImgApp.csv', 'rb')
        nbImgApp = pickle.load(pkl_file)
        pkl_file.close()
    if(os.path.isfile('Resource/nbDim.csv') == True):
        pkl_file = open('Resource/nbDim.csv', 'rb')
        nbDim = pickle.load(pkl_file)
        pkl_file.close()
    if(os.path.isfile('Resource/vecteurPropre.csv') == True):
        pkl_file = open('Resource/vecteurPropre.csv', 'rb')
        vectP = pickle.load(pkl_file)
        pkl_file.close() 
    if(os.path.isfile('Resource/moyenne.csv') == True):
        pkl_file = open('Resource/moyenne.csv', 'rb')
        Moyenne = pickle.load(pkl_file)
        pkl_file.close()
    if(os.path.isfile('Resource/cov.csv') == True):
        pkl_file = open('Resource/cov.csv', 'rb')
        MatriceCov = pickle.load(pkl_file)
        pkl_file.close()
    if(os.path.isfile('Resource/data.csv') == True):
        pkl_file = open('Resource/data.csv', 'rb')
        MatriceData = pickle.load(pkl_file)
        pkl_file.close()
        
# lancement du programme avec menu
while (choixMenu != 0):
    print('Menu  :');
    print('1 - Pretraitement des données via ACP');
    print('2 - Apprentissage avec simplification');
    print('3 - Classification Bayésienne');
    print('4 - Pretraitement + Apprentissage + Classification');
    print('5 - Recherche de la meilleure dimension');
    print('6 - Recherche du meilleur nombre d\'images pour l\'apprentissage');
    print('0 - Quitter');    
    print('');
    choixMenu = input('Selecionner une action (0/1/2/3/4/5/6):');
    print('');

    if (choixMenu == 1):
        
        print ('#### Pretraitement des données via ACP ####')
        print ('')
        print ('\tIndiquer le nombre d\'image pour l\'apprentissage :')
        nbImgApp = input('\tnbImgApp =')
        print (' ')
        print ('==> En cours d\'execution ...');

        vectP,MatriceData = prg.ModulePretraitementACP(nbImgApp)

        print ('==> Pretraitement terminée !');
        print (' ');
        # Suppression des anciens fichier
        if(os.path.isfile('Resource/nbImgApp.csv') == True):
            os.remove('Resource/nbImgApp.csv')
        if(os.path.isfile('Resource/vecteurPropre.csv') == True):
            os.remove('Resource/vecteurPropre.csv')
        if(os.path.isfile('Resource/data.csv') == True):
            os.remove('Resource/data.csv')
        nbDim = 0; # apprentissage obligatoire
        # Sauvegarde    
        output = open('Resource/nbImgApp.csv', 'wb')
        pickle.dump(nbImgApp, output)
        output.close()
        output = open('Resource/vecteurPropre.csv', 'wb')
        pickle.dump(vectP, output)
        output.close()  
        output = open('Resource/data.csv', 'wb')
        pickle.dump(MatriceData, output)
        output.close()
        print ('==> Sauvegarde des fichiers de pretraitement effectué');

    elif (choixMenu == 2):
        if((nbImgApp != 0)and(not(isinstance(vectP, int )))):

            print ('#### Apprentissage avec simplification ####');
            print ('')
            print ('\tIndiquer la dimension pour la projection :')
            nbDim = input('\tnbDim =')
            print (' ')
            print ('==> En cours d\'éxécution ...');

            Moyenne,MatriceCov = prg.ModuleApprentissageAvecAcp(nbImgApp,nbDim,vectP,MatriceData)
            
            print ('==> Apprentissage terminée !');
            print (' ');
            # Suppression des anciens fichier
            if(os.path.isfile('Resource/nbDim.csv') == True):
                os.remove('Resource/nbDim.csv');
            if(os.path.isfile('Resource/Moyenne.csv') == True):
                os.remove('Resource/Moyenne.csv');
            if(os.path.isfile('Resource/cov.csv') == True):
                os.remove('Resource/cov.csv');
            # Sauvegarde    
            output = open('Resource/Moyenne.csv', 'wb')
            pickle.dump(Moyenne, output)
            output.close()
            output = open('Resource/cov.csv', 'wb')
            pickle.dump(MatriceCov, output)
            output.close()
            output = open('Resource/nbDim.csv', 'wb')
            pickle.dump(nbDim, output)
            output.close()
            print ('==> Sauvegarde des fichiers d\'apprentissage effectué');
            
        else:
            print('Faire un pretraitement avant !');

    elif  (choixMenu == 3):

        if(nbDim != 0):

            print ('#### Classification Bayésienne ####');
            print ('')
            print ('\tIndiquer le nombre d\'image a reconaitre :')
            nbImgTest = input('\tnbImgTest =')
            print (' ')
            print ('==> En cours d\'execution ...');

            confusion,tauxErreurGlobal,tauxErreurParClasse = prg.ModuleRecoImg(nbImgTest,nbDim,vectP,Moyenne,MatriceCov)

            print ('==> Classification terminée !');
            print ('');
            print ('Matrice de confusion :');
            print confusion
            print ('');
            for i in range(0,10):
                print ('Taux d\'erreur classe '+str(i)+' = '+str(tauxErreurParClasse[i])+'%')
            print ('->Taux d\'erreur global :');
            print (str(tauxErreurGlobal)+'%');

        else:
            print ('Faire un apprentissage avant !');

    elif (choixMenu == 4):
        
            
        print ('#### Pretraitement avec ACP + Apprentissage avec ACP + Classification ####');
        print ('')
        print ('\tIndiquer le nombre d\'image pour l\'apprentissage :')
        nbImgApp = input('\tnbImgApp =')
        print ('\tIndiquer la dimension pour la projection :')
        nbDim = input('\tnbDim =')
        print ('\tIndiquer le nombre d\'image a reconaitre :')
        nbImgTest = input('\tnbImgTest =')
        print (' ')
        print ('==> En cours d\'execution ...');
        vectP,MatriceData = prg.ModulePretraitementACP(nbImgApp)
        print ('==> Pretraitement terminée !');
        Moyenne,MatriceCov = prg.ModuleApprentissageAvecAcp(nbImgApp,nbDim,vectP,MatriceData)
        print ('==> Apprentissage terminée !');
        confusion,tauxErreurGlobal,tauxErreurParClasse = prg.ModuleRecoImg(nbImgTest,nbDim,vectP,Moyenne,MatriceCov)
        print ('==> Classification terminée !');
        print ('');
        print ('Matrice de confusion :');
        print confusion
        print ('');
        for i in range(0,10):
            print ('Taux d\'erreur classe '+str(i)+' = '+str(tauxErreurParClasse[i])+'%')
        print ('->Taux d\'erreur global :');
        print (str(tauxErreurGlobal)+'%');
        
        # Suppression des anciens fichier
        if(os.path.isfile('Resource/nbImgApp.csv') == True):
            os.remove('Resource/nbImgApp.csv')
        if(os.path.isfile('Resource/vecteurPropre.csv') == True):
            os.remove('Resource/vecteurPropre.csv')
        if(os.path.isfile('Resource/data.csv') == True):
            os.remove('Resource/data.csv')
        if(os.path.isfile('Resource/nbDim.csv') == True):
            os.remove('Resource/nbDim.csv');
        if(os.path.isfile('Resource/Moyenne.csv') == True):
            os.remove('Resource/Moyenne.csv');
        if(os.path.isfile('Resource/cov.csv') == True):
            os.remove('Resource/cov.csv');
        # Sauvegarde    
        output = open('Resource/nbImgApp.csv', 'wb')
        pickle.dump(nbImgApp, output)
        output.close()
        output = open('Resource/vecteurPropre.csv', 'wb')
        pickle.dump(vectP, output)
        output.close()  
        output = open('Resource/data.csv', 'wb')
        pickle.dump(MatriceData, output)
        output.close()    
        output = open('Resource/Moyenne.csv', 'wb')
        pickle.dump(Moyenne, output)
        output.close()
        output = open('Resource/cov.csv', 'wb')
        pickle.dump(MatriceCov, output)
        output.close()
        output = open('Resource/nbDim.csv', 'wb')
        pickle.dump(nbDim, output)
        output.close()
        print ('==> Sauvegarde des fichiers effectué');

    elif (choixMenu == 5):
        #Recherche de la meilleur dimension
        print ('#### Recherche de la meilleure dimension ####')
        print ('\tIndiquer le nombre d\'image pour l\'apprentissage :')
        nbImgApp = input('\tnbImgApp =')
        print ('\tIndiquer la dimension maximale pour la projection (minimum=10) :')
        nbDimMax = input('\tnbDim =')
        print ('\tIndiquer le nombre d\'image a reconaitre :')
        nbImgTest = input('\tnbImgTest =')
        print (' ')
        print ('==> Recherche de la meilleur dimension ...');
        saveTaux = 0.0
        tauxErreurGlobal = 0.0
        vectP,MatriceData = prg.ModulePretraitementACP(nbImgApp)
        print ('==> Pretraitement terminée !');
        listTaux = []
        flag = 0;
        
        r = np.arange(10,nbDimMax+1,1)
        
        for nbDim in r:
            Moyenne,MatriceCov = prg.ModuleApprentissageAvecAcp(nbImgApp,nbDim,vectP,MatriceData)
            print ('==> Apprentissage terminée !');
            confusion,tauxErreurGlobal,tauxErreurParClasse = prg.ModuleRecoImg(nbImgTest,nbDim,vectP,Moyenne,MatriceCov)
            print ('==> Classification terminée !');
            print ('Dimension: '+str(nbDim)+'\t=>\tErreur global: '+str(tauxErreurGlobal))
            listTaux.append(tauxErreurGlobal)
            if (flag == 0):
                saveTaux = tauxErreurGlobal;
                nbDimSave = nbDim
                flag = 1;
            if (tauxErreurGlobal < saveTaux):        
                saveTaux = tauxErreurGlobal
                nbDimSave = nbDim

        print ('->La meilleure dimension trouvé de 10 à '+str(nbDimMax)+' est : '+str(nbDimSave));
        print ('->Erreur Global de : '+str(saveTaux));
        plt.plot(r, listTaux, 'ro')
        plt.show()
    elif (choixMenu == 6):
        print ('\tIndiquer la dimension maximale pour la projection (minimum=10) :')
        nbDim = input('\tnbDim =')
        print ('\tIndiquer le nombre d\'image a reconaitre :')
        nbImgTest = input('\tnbImgTest =')

        saveTaux = 0.0
        tauxErreurGlobal = 0.0
        listTaux = []
        flag = 0;
        
        r = np.arange(1000, 10000, 500)
        
        for nbImgApp in range(1000,10000+500,500):
            vectP,MatriceData = prg.ModulePretraitementACP(nbImgApp)
            Moyenne,MatriceCov = prg.ModuleApprentissageAvecAcp(nbImgApp,nbDim,vectP,MatriceData)
            confusion,tauxErreurGlobal,tt = prg.ModuleRecoImg(nbImgTest,nbDim,vectP,Moyenne,MatriceCov)
            print ('Nombre d\'images pour l\'apprentissage: '+str(nbImgApp)+'\t=>\tErreur global: '+str(tauxErreurGlobal))
            listTaux.append(tauxErreurGlobal)
            if (flag == 0):
                saveTaux = tauxErreurGlobal;
                nbImgAppSave = nbImgApp
                flag = 1;
            if (tauxErreurGlobal < saveTaux):        
                saveTaux = tauxErreurGlobal
                nbImgAppSave = nbImgApp
    
            print ('->La meilleure dimension trouvé de 1000 à 10000 est : '+str(nbImgAppSave));
            print ('->Erreur Global de : '+str(saveTaux));
       
        plt.plot(r, listTaux, 'ro')
        plt.show()