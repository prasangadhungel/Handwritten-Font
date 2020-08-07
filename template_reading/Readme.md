Latest version: 
`read9.py`
This version just works after
`conda activate py36`
TODO:
0. Determine what the relevant qr code "contour"/scanned picture dimensions are.
0.1 In this case, the software does not actually scan the perfect qr code, but it gets a block of 3 vertical blocks, with the top square with the symbol in it being sank down a bit. So you are looking for blocks with aspect ratio 3 (3 high, 1 wide) (+- 20%). 
0.2 Then also determine the width of the entire scan, and compute what the width of the qr codes should be +- 20%. You can compute the width and height of the pixels of the selected contours with aspect ratio 3 by:
0.2.1 The unit of boxHeight (and boxWidth) in the `../template_creating/symbol_spec.txt` file is mm. 
0.2.2 Since you assume A4 paper, which has a page width of .. mm you can compute the image width as n% of the picture width.
0.2.3 You can scan the pixel/image width from python, hence you can compute the image width in pixels in python.
0.2.4 Make 3 scans with 10,20,40,100 boxWidth per line, and verify (measure by hand) what the pixel height of the 3 stacked boxes are (w.r.t. the boxWidth).
0.2.5 Then, compute the bottom 2 boxheights as 2*boxWidth and the heighest box height as =top_box_height=boxWidth*0.9 oid. (Or if that does not fit well to the measured data, make it quadratic.
0.3 Make a margin parameter err_marg= 0.2
0.4 Then only select contours with aspect ratio 3*(1-err_marg)<ar<3*(1+err_marg)
0.5 Then only select countours with:  computed_height*(1-err_marg)< height <computed_height*(1+err_marg)
0.6 Then compute the qrcode field from boxWidth_pixels and simply take the bottom square qr code from the countour.
0.7 Build a classifier that applies a blur.
0.8 Classify black to 1, white to 0. make numpy matrix




0.2.1 Find the width if the image
0.2.2 Read how many images are on each line from `../template_creating/symbol_spec.txt`.
0.2.3 Assumption: The horizontal spacing of the margins left and right on the page, as well as between the images/qr codes is not a function of the nr of rows on a page.
0.2.4 Compute (based on coefficient, (create 3 pdfs with 2, 5,15 images per row) measure the left and right margin width, as well as the horizontal spacing between the images for those 3 settings. 
0.2.4.1 I assume the left margin will be almost constant, approximate its value with width_left_margin=ax+b, where a is the amount of mm/pixels that is added as a function of the number of images per row.
0.2.4.2 The amount of space on the right margin might be a function of pagewith-f(the amount of space on the left margin, pixel width, pixel spacing). A guess is:
width_right_margin=width_left_margin+(n_pictures)*(picture_widht+picture_margin)


0.3 
