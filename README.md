# Therapy Recommendation System using the Item-Based Collaborative Filtering

**UniTn Data Mining Course Project 2021**

At the center of the problem is a dataset of patients. A patient is an entity who has an id, a collection of attributes as {name,value} pairs. These patients are connected to a couple of entities, a condition and a therapy, both having a set of attributes as ¡name,value¿ pairs. A condition is an illness and physicists have designed therapies to treat them. Conditions can be short term or long term. Doctors suggest therapy when a patient is suffering from a condition. Suggested therapy is a trial which is a tuple {t,p,date, params, success}. Here, t is a therapy, p is a patient, date is the time it was applied, and params are pairs of attribute name-value that describe the therapy. When a patient has a condition, a sequence of therapies are suggested as trials which may or may not cure the condition. There is a parameter success which indicates the percentage of the effectiveness of a therapy for a particular condition. This parameter may depend on the history of a patient’s trials. A patient may not have a trial if a condition is detected and not treated yet. We are to design a system to assist therapists suggest an upcoming therapy for a patient for his/her uncured condition. We can summarize that, given the following inputs:

- A set of patients **Pset** with a list of conditions and therapies and trials
- A particular patient and his/her medical history **P**
- An uncured condition Cuncured 
- The output is: A therapy **T**
Moreover, we have to evaluate the accuracy of our output **T**.
#

**Implementations:**
  - [Item-Based Collaborative Filtering](https://github.com/khandakerrahin/patient-treatment-recommender/blob/master/global_similarity_approach.py)
#
**Dataset sources**: 
- [Created a random data generator using **Python**](https://github.com/khandakerrahin/patient-data-generator)
- [Web Scrapping using Python](https://github.com/khandakerrahin/conditionsScrapper)
#

**Detailed Report**: 
- [Link](https://github.com/khandakerrahin/patient-treatment-recommender/blob/master/Report/Therapy_Recommendation_System_using_the_Item_Based_Collaborative_Filtering.pdf)
#
#
