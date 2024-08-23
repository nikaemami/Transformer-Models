# Transformer Models


## 1- Analyzing movie reviews using Transformers

In this project we train a sentiment analysis model using the BERT (Bidirectional Encoder Representations from Transformers) model. Specifically, we parsed movie reviews and classified their sentiment (according to whether they are positive or negative) using the IMDB Moview Reviews dataset. We used the Huggingface transformers library to load a pre-trained BERT model to compute text embeddings, and append this with an RNN model to perform sentiment classification.

<br>

Specifically our model consists of:


• the BERT embedding (whose weights are frozen)


• a bidirectional GRU with 2 layers, with hidden dim 256 and dropout=0.25. 


• a linear layer on top which does binary sentiment classification.


We used the Binary Cross Entropy loss function and the Adam optimizer to train our model. 


<br>

Evaluation on the test dataset:

**Test Loss: 0.231 and the Test Acc: 90.62%**

<br>

Here are some examples and the corresponding predicted scores:


***"Justice League is terrible. I hated it." -> 0.03642885014414787***


***"Avengers was great!!" -> 0.7733321785926819***


***"I think Oppenheimer wasn't as good as people say it is!" -> 0.3038254976272583***


***"I really enjoyed watching La La land. It was perfect!" -> 0.9693074822425842***



<br>

## 2- Vision Transformer

In this project we trained a vision transformer to classify images from the FashionMNIST dataset. We specifically used  a patch size of 4x4, 6 ViT layers, and 4 heads for our ViT.

For doing so, we first convert an input image into a sequence of patches (with patch size equal to 4, as specified), where each patch is embedded into a lower-dimensional space, and then fed into a transformer-based model for image classification. We then implement the multi-head attention mechanism from scratch, and finally define the vision transformer module, using 6 ViT layers and 4 heads, with 10 output classes for the classification task. For this project we used the cross entropy loss function since it is suitable for multi-class classifcation tasks. We also used the Adam optimizer and set the learning rate to 0.001.

We can see the model architecture as below:


<img src="images/3.png" width="600"/>


We visualize the train and test loss curves as below:


<img src="images/4.png" width="600"/>


With these settings, we were able to achieve a final accuracy of 89.33% on our test set.


We plot images with predicted probabilities as below. We can see that the model predicts the true label with high probability.


<img src="images/5.png" width="600"/>


