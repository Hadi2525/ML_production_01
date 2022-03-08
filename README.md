# ML_production_01
This repo is a demo that contains a machine learning model that is used in CI for production.

<h3>
Problem description:
</h3>
In this problem, our goal is to predict the hand writing digits in a picture of 8x8 pixels. 

Here is the summary of the dataset we have available:
<li>
  It consists of 64 features (each feature corresponds to one pixel).
</li>
<li>
  The target is to classify the digits from 0 - 9.
 </li>
 <li>
  The dataset includes 1797 images where each image is represented by its pixel information.
  </li>
 
In order to solve the problem, we build a predictive model using <br> ensemble modeling (random forest classifier)</br> which provides a relative high accuracy model for classification purposes.

But, before implementing the code, we do some exploratory analysis around the data to get a better understanding of what we are dealing with. For that, we follow this procedure:
<li>
  Using the <tt>info</tt> method from <tt>Pandas</tt> to realize what types of features we have.
</li>
