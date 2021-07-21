from load_data import data
from recommender import algo

trainingSet = data.build_full_trainset()

algo.fit(trainingSet)
prediction = algo.predict('E', 2)
rating_of_movie2 = prediction.est
print(rating_of_movie2)