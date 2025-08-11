# Image-Colorizer
A python app using Kwentar's  Color2Embed_pytorch implementation 
https://github.com/Kwentar/Color2Embed_pytorch, the program generates a colorized image using a ref image 
and transforms it from LAB to RGB so that it can be printed via a Gradio interface.

The model operates better with images that are saturated and don't have any background:

<img width="784" height="756" alt="image" src="https://github.com/user-attachments/assets/fe03b1d4-a53a-4124-b71f-d82dc0793d93" />
<img width="784" height="756" alt="image" src="https://github.com/user-attachments/assets/baf706f8-4d58-4ece-a740-c1e6a9955077" />

furthermore images that are greenish/blue tent to perform worse, ajusting color bias can help.


## Instalation:

Withing the readme there is instructions on what pything libraries must be installed + links to the model weight (i can't add it to github due to size)
