# Présentation du cours

Dans le cadre du cours Gestion et analyse de données, nous ont été présenté et confronté différentes méthodes de modélisation de données : les entrepôts de données et les systèmes de traitement analytique en ligne (OLAP), les bases de données non-relationnelles (NoSQL) et le traitement des données massives, les données transactionnelles et la recherche de règles d'associations. De plus, nous avons étudié l'extraction de connaissance et le forage de données, puis les techniques d'exploitation de données prédictives et descriptives.

# Description du travail

Mots clés: Apprentissage machine, Convolutional neural network, Intelligence artificielle

Le but de ce travail est de réaliser un système capable de reconnaître des champignons selon leur espèce, en utilisant leurs images, et de déterminer s'ils sont comestibles ou pas.

Pour arrivé à cette fin, il faudra réaliser un réseau de neurones de type CNN sans utiliser un réseau pré-entraîné. Il faudra déterminer l’architecture ainsi que les hyperparamètres à donner à ce réseau, soit : 
- la taille de l'image
- l’architecture du réseau (taille des couches, types de couches, combien de couches)
- la taille des batchs, epochs et itérations
- Le learning rate
- L’utilisation ou non d’un optimiseur
- La modification du dataset

# Les technologies/APIs utilisées

Python 3, Jupyter Notebook, Keras

# Rapport

Par Yacine Sehboub, en date du vendredi 17 décembre 2021.

Tout au long, nous utilisons 80% des échantillons pour le training, et 20% pour les tests.

Nous effectuons des tests d’overfitting. Un test positif donne une accuracy un peu plus petite et un loss un peu plus grand. Des valeurs trop proches ou trop loin sont signes d’overfitting.

## Modèle #1

Architecture et characteristiques:
1. Compression d’image 64*64*3
2. Convultion
3. Pooling
4. Flattering
5. Batch = 25
6. 1 couche cache avec 32 neuronnes
7. 10 neurrones de sorties avec la fonction d'activation softmax
8. Optimizer adam
9. Pre-processing des images pour diminuer l'overfitting
10. 5 Epochs
11. Accuracy: 0.2650
12. Loss: 2.0779

Test d’overfitting:

Accuracy = 0.2599, Loss = 2.1468

Delta : correct

## Modèle #2

On a essayé de normaliser les photos d’apprentissage vue que les modèles apprennent mieux avec des valeurs basses. Mais cela n’a donné aucun résultat.

*entrainement = tf.keras.utils.normalize(entrainement, axis=1).reshape(entrainement.shape[0], -1)*

Puis nous avons essayé d’augmenter la qualité des images pour 128pi au lieu de 64pi. Seulement, cela a affecté négativement l’apprentissage. Puis nous avons diminué la qualité pour 32pi.

Architecture et characteristiques:
1. Compression d’image 32*32*3
2. Convultion
3. Pooling
4. Flattering
5. Batch = 25
6. 1 couche cache avec 32 neuronnes
7. 10 neurrones de sorties avec la fonction d'activation softmax
8. Optimizer adam
9. Pre-processing des images pour diminuer l'overfitting
10. 5 Epochs
11. Accuracy: 0.1125
12. Loss: 2.2827

Test d’overfitting:

Accuracy = 0.1099, Loss = 2.2718

Delta : correct

## Modèle #3

Nous sommes revenus à 64 pixels. Cette fois, nous avons augmenté la taille des couches cachées, ainsi que leur nombre. Cela a augmenté la précision. Nous avons ensuite essayé de passer à 3 batch, par curiosité, et cela a grandement affecté l’apprentissage de manière négative. Nous avons donc essayé de faire passer le nombre de batch à 50, mais à notre grand malheur, cela provoque des problèmes dans le code – Nous avons donc dû abandonner cette approche.

Architecture et characteristiques:
1. Compression d’image 64*64*3
2. Convultion
3. Pooling
4. Flattering
5. Batch = 25
6. 3 couche cache avec 128 neuronnes
7. 10 neurrones de sorties avec la fonction d'activation softmax
8. Optimizer adam
9. Pre-processing des images pour diminuer l'overfitting
10. 5 Epochs
11. Accuracy: 0.1125
12. Loss: 2.2827

Test d’overfitting:

Accuracy = 0.1045, Loss = 2.1689

Delta : correct

## Modèle #4
Pour ce modèle, nous nous sommes d’abord intéressé au nombre d’Epochs. Nous avons donc essayé d’augmenter ce nombre à 25. Et après un très long et stressant apprentissage, nous obtenons une accuracy totalement impressionnante… mais nous ne sommes pas dupes, nous soupçonnons de l’overfitting.

Architecture et characteristiques:
1. Compression d’image 64*64*3
2. Convultion
3. Pooling
4. Flattering
5. Batch = 25
6. 3 couche cache avec 128 neuronnes
7. 10 neurrones de sorties avec la fonction d'activation softmax
8. Optimizer adam
9. Pre-processing des images pour diminuer l'overfitting
10. 25 Epochs
11. Accuracy: 0.7788
12. Loss: 0.8801

Test d’overfitting:

Accuracy = 0.2949, Loss = 3.867

Delta : trop grand, **overfitting**

## Modèle #5
Pour ce dernier modèle, nous avons décidé d’augmenter le nombre d’Epochs, mais juste un peu. Puis nous avons ajouté 25 photos pour chaque classe. Finalement, nous avons essayé la fonction d’optimisation ‘opt’ au lieu de ‘adam’. Seule la dernière modification fut négative.

Architecture et characteristiques:
1. Compression d’image 64*64*3
2. Convultion
3. Pooling
4. Flattering
5. Batch = 25
6. 3 couche cache avec 128 neuronnes
7. 10 neurrones de sorties avec la fonction d'activation softmax
8. Optimizer adam
9. Pre-processing des images pour diminuer l'overfitting
10. 10 Epochs
11. Accuracy: 0.3409
12. Loss: 1.247

Test d’overfitting:

Accuracy = 0.3299, Loss = 1.832

Delta : correct

# Référence 

Les 10 espèces de champignons et leurs caractères de comestibilité

| Espèce de champignon        | Comestibilité  |          
| --------------------------- | -------------- |
| Ganoderma pferfferi         | comestible     |
| Pluteus cervinus            | comestible     | 
| Plicatura crispa            | non-comestible |  
| Tricholoma scalpturatum     | comestible     |
| Xerocomellus chrysenteron   | comestible     | 
| Armillaria lutea            | non-comestible |
| Mycena galericulata         | comestible     |
| Coprinellus micaceus        | non-comestible |      
| zebra stripes               | are neat       |
| Fomes formentarius          | non-comestible |
| Fomitopsis pinicola         | non-comestible |

# Sources

Auteur de l'énoncé du travail: Jean Massardi ( linkedin.com/in/jean-massardi-phd-89359118 )

Sources des images: https://www.kaggle.com/c/fungi-challenge-fgvc-2018/overview

Inspiration pour le premier modèle: https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

