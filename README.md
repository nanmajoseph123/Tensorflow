# Tensorflow

Module1:Complete Guide Of Installation Of Tensorflow
with code examples

SRH UNIVERSITY OF APPLIED SCIENCES
Professor: Mr.Alexander Iliev

***************************************************************************Tensorflow 2.0*******************************************************************************************************************


Introduction:Tensorflow is a popular framework developed by Google. It is an open source platform for different Operating Systems.It can have any sequence of operation. It makes the data acquisation process, data predictions methodology relatively easier. It helps in churning out narrower results. Everything is done in Python which provides a user friendly API for building applications with framework to be later executed in C++.It could be extensively used to run deep neural networks for image processing, data acquisition, and a lot of other simulations.


 
Course Description:Complete guide of Installation of Tensorflow with code examples and expansion of research in different environment and scenarios.This course include the presentation ,media files, module.py files and readme.txt files.


Prerequisites:

Need to install Tensorflow with 2.0.0 latest version.
1.) Installing Anaconda
2.) Installing different Python versions
3.) Installing Tensorflow for different environments and checking the platform independencies.
4.) Various Platforms like
*  Windows
* Mac OS
* Linux
* Raspbian

****************************************************************************Installation********************************************************************************************************************





*********************************************1.)Windows OS***********************************************************************

¥ Install Stable version of Python(python.org/downloads/release/python-374/)

¥ Tick Custom Installation (Preferred) or in Manual Installation ensure ÒpipÓ is chosen as the optional feature in the path environmental Variable.
¥ Alternatively you can install PiP  Dependencies via Command Prompt.
¥          c:\>pip3 install six numpy wheel
¥          c:\> pip3 install keras_applications==1.0.6 --no-deps
¥          c:\> install keras_preprocessing==1.0.5 --no-deps

     Install Bezel :
¥ Specially recommended for 64 Bits Windows 10
¥ Install Pre-Requisites : 
¥ Visual C++ Redistributable for Visual Studio 2015
¥ Msys2 x86_64
¥ after Msys2 Installation ,run following command in Msys2 Prompt  
¥ -pacman ÐS zip unzip patch diffutils git 

     
     Download Bezel:
¥ choose Windows Ðx86_64.exe file 
¥ Setup Bazel to build C++

¥ Install Visual C++ Build Tool 2017:
             - Download Visual Studio 2017
Download and Install :

¥ Microsoft Visual c++ 2017 Redistributable
¥ Microsoft Build Tools 2017 
 

         
Install GPU:
¥ In the command Prompt, type :
            - c:\> git clone https://github.com/tensorflow/tensorflow.git
            - c:\> cd tensorflow
            -git checkout branch_name  # r1.9, r1.10, etc. ( r1.15 will be the last 1.x release for tensorflow.It will be followed by  patches and small updates)


Configure the Build:
¥ Configure the build of your system by running the following code at the source of your TensorFlow

      C:\>python ./configure.py
  
Build PiP Package
¥ For CPU:
             C:\>bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

¥ For GPU :
           Tensorflow PiP packages will by include GPU support. It will work on machines with and   without Nvidia GpuÕs.

Build the Package:

¥ Bazel build is a command that creates an executable command- build_pip_package

 c:\> bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg

Install the Package:
     
¥ Use the pip3 install command to install the package. 
             
c:\>pip3 install C:/tmp/tensorflow_pkg/tensorflow-version-cp36-cp36m-win_amd64.whl




*********************************************2.) UBUNTU:************************************************************************

¥ In the command Terminal type :

$sudo apt install python-dev python-pip  # or python3-dev python3-pip  
  
      -As for the Windows Os,we need to install TensorFlow PiP package dependencies. For people with virtual environment omit the  --user argument in the code.

- $ pip install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1Õ
- $ pip install -U --user keras_applications==1.0.6 --no-deps
- $ pip install -U --user keras_preprocessing==1.0.5 --no-deps

¥ Download tensorflow Source code:

           - $  git clone https://github.com/tensorflow/tensorflow.git
           - $  cd tensorflow 
¥ To configure the build :

            -  $  ./configure


¥ Build the PiP Package
      Installing Bazel :
      Bazel is supported over 2 platforms
      - 18.04(LTS)
     -  16.04(LTS)


Using Required Packages:

$ sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3     #Pre-requisities


¥ Download Bazel

¥ Same as Windows, we need to download Bazel binary installer name 

          bazel-<vesion>-installer-linux-x86_64.sh from Bazel Build.



¥ Run the installer Package: Run the following command
            chmod +x bazel-<version>-installer-linux-x86_64.sh
            ./bazel-<version>-installer-linux-x86_64.sh Ðuser


¥ Add Bazel to Environmental Path Variable

export PATH= Ò$PATH : $HOME/gitÓ


¥ We can build JAVA code using Bazel, by installing a JDK

$ sudo apt-get install open jdk-8-jdk   #16.04(LTS)
$ sudo apt-get install open jdk-11-jdk  # 18.04(LTS)


¥ Add bazel distribution URI as a package source

echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt-get install curl
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add Ð





¥ Install and update Bazel 
$  sudo apt-get update && sudo apt-get install bazel    #install
$   sudo apt-get install --only-upgrade bazel                     #update
Bazel Build for TensorFlow Packages
$ bazel build //tensorflow/tools/pip_package:build_pip_package


       
¥ We can use Bazel to make the tensorflow package builder with only CPU support.
     $  bazel build --config=opt 


//tensorflow/tools/pip_package:build_pip_package


¥ GPU support :
         bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package


¥ Build Package:
       The Bazel command which we used previously created and executable file names build-pip-pack. This program helps building the PiP package.

¥ From Release branch:

- $  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg 

¥ From Master Branch:

- $  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg


¥ Install Package:
- $  pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl


¥ Tensorflow PiP dependencies :

 $  pip install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1Õ
   
   
    $  pip install -U --user keras_applications==1.0.6 --no-deps
 
    
    $   pip install -U --user keras_preprocessing==1.0.5 --no-deps



*******************************************************3.)MAC OS**************************************************************************


¥ Python version by default on the mac system 2.7
¥ Check python version by 

   Python --version

¥ Python version after installing anaconda  3.7.3

¥ Created the virtual environment with python 3.7 for       

  Tensorflow 2.0.0 version.

¥ Virtual environment created:-

   Source activate tensorflow


¥ Installed Juypter:- 

   conda install Juypter.

¥ Installing Scipy:-

conda install scipy

¥ Installing Sklearn :- 

pip install --upgrade sklearn
  
¥ Installing Panda:-   
 
   pip install -Ðupgrade pandas

   Installing Panda-Datareader:-
   
   pip install -Ðupgrade panda-datareader

¥ Installing Matplotlib:- 

   pip install -Ðupgrade matplotlib



¥ Installing Pillow


pip install Ðupgrade pillow

¥ Installing requests


pip install -Ðupgrade requests


¥ Installing  h5py


   pip install -Ðupgrade h5py


¥ Installing psutil


pip install -Ðupgrade psutil

¥ Upgrading tensorflow


pip install -Ðupgrade tensorflow==2.0.0


¥ Installing Keras


pip install -Ðupgrade Keras=2.2.5
   python
   import tensorflow as tf
   print(tf.__version__)

   

 python -m ipykernal install Ðuser --name tensorflow Ð-display-name ÒPython 3.7(tensorflow)Ó 

¥ Move the program folder to the home directory
¥ Exit the terminal window
¥ Reopen the terminal window
¥ Type to open the working directory 
pwd
¥ Activate the virtual environment
Source activate tensorflow
¥ Type the course folder name
Cd programs/






*******************************************************4.) Raspbian *******************************************************************


The repository defaults to Master build. For release branch build
$ git checkout branch_name      # r1.9, r1.10, etc.

Cross-compile the TensorFlow source code to build a Python pip package with ARMv7 NEON instructions that works on Raspberry Pi 2 and 3 devices. 




$   CI_DOCKER_EXTRA_PARAMS="-e CI_BUILD_PYTHON=python3 -e        	CROSSTOOL_PYTHON_INCLUDE_PATH=/usr/include/python3.4" 
\tensorflow/tools/ci_build/ci_build.sh PI-	PYTHON3 \tensorflow/tools/ci_build/pi/build_raspberry_pi.sh       #Python 3.0
$  tensorflow/tools/ci_build/ci_build.sh PI \tensorflow/tools/ci_build/pi/build_raspberry_pi.sh  #Python 2.7


Can be installed using CONDA or PiP.

conda install -c conda-forge jupyterlab  #CONDA

pip install jupyterlab    #PiP




********************************Built with******************************************************


Frameworks used:

Tensorflow





************************Versioning**************************************************************


Python 3.7
Tensorflow 2.0.0





***********************Acknowledgements*********************************************************




https://www.slideshare.net/matthiasfeys/introduction-to-tensorflow-66591270
https://www.slideshare.net/Ahmedrebai2/tensorflow-presentation
https://www.infoworld.com/article/3278008/what-is-tensorflow-the-machine-learning-library-explained.html
https://www.toptal.com/machine-learning/machine-learning-theory-an-introductory-primer
https://monkeylearn.com/blog/gentle-guide-to-machine-learning/
http://www.cs.stir.ac.uk/~lss/NNIntro/InvSlides.html
https://towardsdatascience.com/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf
https://iamtrask.github.io/2015/07/12/basic-python-network/https://www.guru99.com/download-install-tensorflow.html
https://www.guru99.com/download-install-tensorflow.html
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_operations.py
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb#scrollTo=_TZTwG7nhm0C
https://github.com/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb
https://www.google.com/search?q=a+perfect+ppt+presentation+on+tensorflow+installation&oq=a+perfect+ppt+presentation+on+tensorflow+installation&aqs=chrome..69i57j69i64.28735j0j7&sourceid=chrome&ie=UTF-8
https://de.slideshare.net/matthiasfeys/introduction-to-tensorflow-66591270
https://www.tutorialspoint.com/tensorflow/tensorflow_tutorial.pdf
https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf
https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
https://hub.packtpub.com/tensorflow-always-tops-machine-learning-artificial-intelligence-tool-surveys/
https://venturebeat.com/2018/10/24/airbnb-details-its-journey-to-ai-powered-search/
https://medium.com/tensorflow/intelligent-scanning-using-deep-learning-for-mri-36dd620882c4
https://www.airbus.com/newsroom/news/en/2016/12/Artificial-Intelligence.html
https://www.qualcomm.com/news/onq/2017/01/09/tensorflow-machine-learning-now-optimized-snapdragon-835-and-hexagon-682-dsp
https://machinelearningmastery.com/introduction-python-deep-learning-library-tensorflow/
https://www.digitaldoughnut.com/articles/2017/march/top-5-use-cases-of-tensorflow
https://www.tensorflow.org/guide/data#basic_mechanics
https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/architecture.md
https://www.tensorflow.org/guide/effective_tf2
https://www.tensorflow.org/guide/keras/overview
https://en.wikipedia.org/wiki/Keras
https://hub.packtpub.com/top-10-deep-learning-frameworks/
https://en.wikipedia.org/wiki/PyTorch
https://malmaud.github.io/tfdocs/
https://www.youtube.com/watch?v=mcIKDJYeyFY
https://www.youtube.com/watch?v=SNdQqYpfCV4
https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python/
https://www.tensorflow.org/install
https://gist.github.com/PurpleBooth/109311bb0361f32d87a2




**************************************************************************************THE END******************************************************************************************************

