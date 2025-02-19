# Wildfire-Risk-Assessment-Model
Entered this project in the Sacramento State AI Hackathon winning $500 for finishing in 2nd place out of 16 teams.

FULL DESCRIPTION:
Formed a group of 4 other like-minded students and entered this project in the AI Hackathon at the Carlsen Center for Innovation and Entrepreneurship sponsored by California State University, Sacramento. We won $500 for finishing in 2nd place out of 16 teams.

Our group, the "Round Table Rascals," developed an AI model that focused on wildfire prevention by identifying vulnerable areas that require proactive measures for local governments, firefighters, and at-risk communities.

WRAM uses satellite data to predict areas at risk of wildfires. We analyze images from Sentinel-2 satellites, focusing on pre-fire conditions. Each image is broken into 12 distinct spectral bands, which help identify the environmental factors of a fire starting, such as the levels of moisture in the air or certain light waves being present. 

On Hugging Face, we found a hand labeled mask created by the California Department of Forestry and Fire Protection identifying the locations where a forest fire was known to have taken place. We used Google Colab along with Jupyter Notebooks to analyze and visualize the data, helping us understand patterns related to wildfire-prone areas. We found that the dataset was highly imbalanced with many images containing less that 0.01% fire. We chose to patch our 5490x5490 images into 512x512 patches, and oversample patches which represented fire to address this imbalance.

Our model is a Convolutional Neural Network (CNN) based on U-Net architecture, optimized for image segmentation. During training, we compared the predicted wildfire risk mask against a ground truth mask (hand labeled images), representing the areas known to have been affected by wildfire to calculate loss, using this to refine model accuracy. The model generated a mask of predicted high-risk areas with 92% accuracy.

Looking ahead, we aim to partner with local governments and the California Department of Forestry to integrate WRAM into early warning systems and develop a public-friendly interface.
