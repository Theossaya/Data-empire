
# Movie Rating Prediction with AutoEncoders

## Project Description

This project focuses on predicting movie ratings using AutoEncoders, a type of neural network designed to learn efficient representations of data, typically for the task of dimensionality reduction. Using the MovieLens dataset, which comprises over a million ratings from thousands of users, we aim to create a model that can predict how a user would rate a movie they haven't seen based on their rating history.

## Data Description

The dataset is the  MovieLens 100k and it contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users. It includes user demographic information and movie details as follows:

- Ratings: Range from 1 to 5 stars, with each user having at least 20 ratings.
- Users: Identified by unique IDs with associated gender, age group, occupation, and zip code.
- Movies: Identified by unique IDs with associated titles, release years, and genres.

### Data files:

- ratings.dat: UserID::MovieID::Rating::Timestamp
- users.dat: UserID::Gender::Age::Occupation::Zip-code
- movies.dat: MovieID::Title::Genres
The data has been pre-processed to format the user IDs, movie IDs, and ratings into a matrix where rows correspond to users and columns correspond to movies.

### Pre-processing Steps

Pre-processing involves:

- Converting the sparse rating matrix to a user-movie dense matrix.
- Handling missing values.
- Converting data into PyTorch tensors.

## Model Architecture

The AutoEncoder architecture implemented in this project consists of:

- An input layer with the number of neurons equal to the number of movies.
- Two hidden layers that encode and decode the ratings respectively.
    - Hidden layer 1: 20 neurons with Sigmoid activation.
    - Hidden layer 2: 10 neurons with Sigmoid activation.
- An output layer umber of neurons equal to nb_movies, using Sigmoid activation for reconstruction.
The model uses Sigmoid activation functions, the RMSprop optimization algorithm, and Mean Squared Error (MSE) for the loss function.

## Training and Evaluation

The model was trained over 200 epochs. The training process involved feeding the input through the network, comparing the output with the target, computing the loss, and adjusting the weights of the model accordingly.

The performance was evaluated based on the test set loss, calculated only on the ratings that existed in the test set to ensure a fair assessment of the model's predictive capability.

## Results and Insights

Upon training, the model achieved a test loss of 79% indicating the model's effectiveness at reconstructing the input ratings. Insights into the model's performance and user rating patterns were obtained, which could be used to improve recommendation systems.

## Code

The project's implementation is documented in a Jupyter Notebook, with clear explanations of each step involved in the process, from downloading and pre-processing the data to training and evaluating the model.

### Files included:

- AutoEncoders.ipynb: The main notebook containing the full pipeline of the project.


