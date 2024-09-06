This is my first attempt at a neural network. It was built from scratch just using numpy and standard math libraries. 


If you want to test this code for yourself, you'll need to install python, and you'll likely need to use the pip tool to download some libraries.


To test the network, You'll want to run digitRecognizerDrawer.py. This brings up a separate window where you can draw digits. Click ENTER/RETURN to have the system guess whats written and press C to clear the canvas.


NOTE: 

You may notice that the guesses are pretty bad. This network had like 93 percent accuracy on sample test MNIST data but it performs terribly with the program I made. It needs something called "normalization" which basically means that the network was trained with a 
certain format in mind, and I didn't "normalize" my data to this standard. It's like if you only ever read books and never read actual handwriting, you would struggle to read someone's less-than-perfect handwriting. There could be other issues, but that seems to be the
main one. When I return to this project that's the first thing I'll address.
