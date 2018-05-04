---
title: MBTA Rider Segmentation
---

## Contents
{:.no_toc}
*  
{: toc}

## Team
Harvard 2018 Spring AC297r Capstone Project: Chia Chi (Michelle) Ho, Yijun Shen, Jiawen Tong, Anthony Hou

## Motivation & Problem Statement  

The Massachusetts Bay Transportation Authority (MBTA) is the largest public transportation agency in New England, delivering a complex system of subway, bus, commuter rail, light rail, and ferry services to riders in the dynamic economy of the Greater Boston Area. It is estimated that MBTA provides over 1.3 million trips on an average weekday. 
	While MBTA collects a wealth of trip transaction data on a daily basis, a persistent limitation has been the organization’s lack of knowledge around rider groups and their respective ridership habits. Understanding rider segmentation in the context of pattern-of-use has significant implications in developing new policies to improve its service planning and potentially changing its fare structure. Therefore, we aim to develop a flexible, repeatable rider segmentation model on MBTA’s “core system” (encompassing local buses and subway) that can group individuals according to pattern-of-use dimensions. 


## Project Deliverables
Our project deliverables are:

- A reusable Python segmentation package that clusters riders and infers rider group characteristics.
- A web-based visualization exploration tool intended to facilitate the discovery of business and ridership insights.

| <img src="img/project_deliverables.png" width="1000">| 
|:--:| 
| *Figure 1: Project Deliverables* |

The specific goals of each project deliverables are 
- For the Python segmentation package:
    - Develop a method to extract rider-level pattern-of-use features from transaction data
    - Develop a method to cluster riders using unsupervised learning algorithms based on extracted features
    - Develop a method to profiles rider clusters using demographic information
    - Develop a generative model that automatically generates simple reports describing rider clusters
    - Implement simple static visualization functions to display various rider segment characteristics

- For the visualization exploration tool:
    - Implement a fully dynamic web-based application with a Flask backend to display interactive D3 visualizations for data exploration (App with full functionality, not deployed to the web)
    - Implement a static version of the full app that displays the same interactive D3 visualizations without a Flask backend (App with limited functionality, deployed using Github pages)

## Data Description
Available data: 
- MBTA Automated Fare Collection (AFC) data containing transaction-level data
- MBTA Fare product data containing fare product definitions
- MIT/MBTA ODX data containing transaction-level origin/destination inference
- MBTA Stops data containing information about each station/stop
- GoogleMap Geoencoding API, which maps longitude and latitude coordinates to zip codes 
- US Census data containing demographics information by zip codes





## Literature Review

## Modeling Approach

## Sample Results
- Compare hierarchical vs. non-hierarchical
- Compare lda vs. kmeans

## Conclusions

## Future Work
