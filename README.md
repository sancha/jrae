#Java Recursive Autoencoder

jrae is a re-implemention of semi-supervised recursive autoencoder in java. This package also contains code to demonstrate its usage.  

More details are available at 

http://www.socher.org/index.php/Main/Semi-SupervisedRecursiveAutoencodersForPredictingSentimentDistributions

In short, semi-supervised recursive autoencoder is a feature learning algorithm to learn an encoding for text data and that can then be used for performing classification. The jrae package is pretty comprehensive - it includes code for learning the features as well as for performing basic classification, and is parallelized to run on a multi-core machine.

The package includes a demo of movie review classification on which the algorithm attains state-of-art results.

Downloading

The Recursive Autoencoder code is being maintained on github and can be 
downloaded at

  https://github.com/sancha/jrae 

#Dependencies

The RAE package requires the jblas package for supporting the linear algebra 
operations. These requirements are included in the lib directory.

* jblas 
* junit4
* log4j
* jmatio

Including the jblas jar file may not be sufficient. JBLAS requires either
LAPACK or ATLAS. Check out https://github.com/mikiobraun/jblas if you run 
into trouble. If you are running ubuntu, do `sudo apt-get install 
libgfortran3`.

#BUGS

If you encounter any bugs, please report it on github.

* Author: Sanjeev Satheesh <ssanjeev@stanford.edu>
* Created: 2012 February 20
* Keywords: java, sentiment analysis, machine learning, nlp 
* URL: <http://github.com/sancha/jrae>

#WORD2VEC

The core feature of the recursive autoencoder is to learn a representation of words and sentences. Google recently released a similar tool, you are encouraged to try out the word2vec project http://code.google.com/p/word2vec/ 
