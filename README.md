What is this application?
=========================
It's a handwritten digit detector with a webcam in a noisy environment 
(my 2 years old daughter's notebook) deployed on a Raspberry Pi 3 Model B.

How this application works?
===========================
Under the hood there is a denoising algorithm based on the total
variation minimization of the captured image and a convolutional neural network 
trained and deployed with Keras.

Raspberry Pi configuration
==========================

The first thing to do is to install an operating system in your memory card
SD. For that purpose you need to: 

* 	Format and create a partition of your SD card. 
	In my case I was using Windows on my computer so I recommend 
	a free edition of **EaseUS Partition Master** 
	<https://www.easeus.fr/partition-manager/index.html>
* 	Install an image of your prefered OS in the SD card. 
	In my case I 
	downloaded an Ubuntu MATE image for Raspberry Pi 
	from <https://www.raspberrypi.org/downloads/> 
	and then I used **Win 32 Disk Imager** to write the disk image
	to the SD card.    
* 	Insert your SD card into your
	Raspberry Pi and connect also a mouse and a screen. Then connect 
	the energy
	power cable. The OS will be launched automatically for the first time. 
	Follow the OS 
	configuration process with the help of the screen keyboard (Go to 
	Activities ->
	Settings -> Universal Access -> Switch to screen keyboard in the 
	typing section).
*	Activate automatic reconnection (i.e. no need to introduce the 
	password of your account) if you think 
	you will be the only user or just temporarely while carrying out 
	the setup. This will help you to avoid using
	for each connection the Ubuntu screen keyboard.
	
Once you have finished with that, the next step is to connect remotely to 
your Raspeberry Pi from your computer (connected to the same private local 
network). For that purpose:	
	
* 	Connect to your home Wifi connection. It may occur that your Wifi 
	connection is not visible. 
	This is probably due to the fact that your router is configured to use 
	signal channels that are not allowed in the US (this was my case in France
	with channels above 11). 
	To solve this problem, do
	```console    
	$ sudo nano /etc/default/crda
	```
	and make the last line to read `REGDOMAIN=FR` to change the country 
	permanently. Save and close the editor.
*	Install and activate the SSH service in your Raspeberry Pi: 
	```console    
	$ sudo apt install openssh-server
	$ sudo systemctl enable ssh.service
	$ sudo systemctl start ssh.service
	$ sudo dpkg-reconfigure openssh-server
	```
	and create a pair of public/private RSA keys:
	```console    
	$ /usr/bin/ssh-keygen -A
	```
	You can check if the ssh connection works fine from your computer
	with a ssh client such as **PuTTY** or **MobaXterm**.
*	Install Real VNC to control the Raspberry Pi's screen remotely 
	from your computer. This is essential for telemetry applications
	(such as this one) where the webcam plays an important role.
    To install RealVNC Server in Ubuntu, you need to use 
	`apt-get install`: 
	```console    
	$ sudo apt install realvnc-vnc-server realvnc-vnc-viewer
	```
	however this option didn't work for me. You can also  
	download the sources from here 
	<https://www.realvnc.com/fr/connect/download/vnc/raspberrypi/>
	to <path/to/file>
	and then install them through the console:
	```console    
	$ sudo dpkg -i <path/to/file>
	```
    Then enable the correct service and start it:
    ```console    
    $ sudo systemctl enable vncserver-x11-serviced.service
    $ sudo systemctl start <service>
    ```
*   Or enable VNC Server at the command line using raspi-config:
    ```console    
    $ sudo sudo raspi-config
    ```
    and then navigate to interfacing options, scroll down and select
    VNC > Yes.
*	Get your IP address in your private local network:
	```console    
	$ ifconfig -a
	```
*   Download and install the VNC Viewer on your computer. Then enter your 
    Raspberry Pi's private IP address into VNC Viewer. You are now connected
    remotly to your Raspberry Pi!


Python packages installation
============================

Installing the right Python packages in a Raspberry Pi processor 
architecture (ARM) can be quite
challeging. Here below a few reasons why:

* 	If you try to use Conda as a package and virtual enviroment manager
    please don't. You will be limited to Python 3.4 at most and the 
    available packages will also be limited. While on a PC Anaconda is a good 
    way to get an 'up-to-date' python with rich repositories, 
    it's not so great for Raspberry Pi.
* 	If you try to install them using `pip` you will soon realize
    that many packages will need to be build from source since no 
    wheels (built distributions) are available. 
    The main reason is that certain libraries (such as
    SciPy) rely in compiled C and fortran libraries, that need to be
    compiled in the same architecture. Usually `pip install` would fetch
    prebuilt packages but the Raspberry Pi's ARM architecture is not 
    really supported. The default behavior of `pip` is to always prefer
    wheels because the installation is always faster.
    Building a package (i.e. composing compiling and linking) from source 
    is a processor architecture and OS dependent procedure 
    that can be quite slow and annoying due to the many dependencies potential
    errors.
* 	One could just then build from source some packages but there
    is an additional problem: the small amount of RAM memory available
    in the Raspberry Pi.

The solution that I found to deal with the above problems 
was to install as much as possible 
packages using `pip3 install` and then rest of them, for those whom the
build process wasn't working or taking too much time (as it was the case
for `OpenCV` and `SciPy`), run in the console:
```console    
$ sudo apt-get install python3-pkg_name
```    
as well as other required packages such as a BLAS/LAPACK math
library with development headers, e.g. `libopenblas-base`, 
`libopenblas-dev`, `python-dev`, etc.  

**Remark**
Both `apt-get` and `pip` are mature package managers which automatically 
install any other package dependency while installing. `pip` has the 
advantages of letting you chose various python package versions or to
install a package in a virtual environment or install a package which 
is only hosted on PyPI. `apt-get` on the other side proposes only one
pre-compiled version of a python package.