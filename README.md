Melody Generation from Lyrics with Local Interpretability
===============================================================
Wei Duan, Yi Yu, Xulong Zhang, Shuhua Tang, Wei Li, Keizo Oyama
===============================================================
Digital Content and Media Sciences Research Division, National
Institute of Informatics, SOKENDAI, Japan
===============================================================

Where to find the code ?

    Code related to paper can be found at ./code/gan

        train.py is be used to train the model
        evaluate.py is used to evaluate the model on test data
        generate.py is used to generate melody for a given lyrics
        
    Code to produce visualisations can be found at ./code/viz

        generated figures can be found at ./code/viz/figures

How to generate a melody for a given lyrics?

    Use the script generate.py found at ./code/gan
    
    To generate a melody you need to pass
    
        --SYLL_LYRICS : it corresponds to the syllable lyrics
        --WORD_LYRICS : it corresponds to the word lyrics 
        --CKPT_PATH   : it corresponds to model checkpoint path
        --IS_GAN      : set to generate using GAN
        --MIDI_NAME   : (optional) name of generated midi 
        
     refer to ./code/gan/run.ipynb for usage.
        
How to change model hyper-parameters ? 

    Hyper-parameters can be changed by editing the settings.txt (json) file.
        
        It can be found at ./code/gan/settings
     
Where to find model checkpoints ?

    Checkpoints for MLE can be found at ./checkpoints/gan/pre_train_gan
    Checkpoints for GAN can be found at ./checkpoints/gan/adv_train_gan
    
Where to find the evaluation result ?

    Results for MLE & GAN can be found at ./results/gan/
  
How to produce lyrics-synthesized melody ?

    We use Synthesizer V with the voice of Eleanor Forte.
    
