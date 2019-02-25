# Plan Your Meetup

### Summary:

**Plan Your Meetup** is a web service that helps novice hosts better plan their events, by providing various insights about meetup scheduling, publicity ideas, and recommendations.

### Dataset:

The dataset was fetched from the [Meetup API][1] utilizing a [python API][2] client. In particular, We first identified the Meetup groups(>100 members) in the top 30 US cities with the most number of groups and then gathered the meetup details associated with these groups between 2011 and 2018 July. The raw dataset is around 10G, and we use PySpark to implement [distributed data analysis][12].

This app includes three modules:

#### 1. [Meetup Occurrence][4] 

When do meetups happen? And, what is the best time to schedule meetups? Here, we used [exploratory data analysis][3] to reveal the daily and hourly occurrence of meetup events and generated [visualizations][5] with Bokeh.

Meetup hosts can use this insight to schedule their events.

#### 2. [Topic classifier][9] 

We trained [machine learning][6] and [deep learning][7] models to classify the topics of meetup descriptions. [Benchmark testing][8] with cross-validation indicated that a GRU model showed the best performance.

Meetup Hosts can use [word clouds][10], which present the most important keywords, to generate ideas for writing their meetup descriptions.

#### 3. Venue Recommender

We are building a hybrid recommender system which combined content and collaborative filtering. This part is ongoing, see [here][11] for an exploration of spatial distributions of popular venues.

This app could help organizers explore fantastic venue options and get recommendations favored by other hosts.

[1]:https://www.meetup.com/meetup_api/
[2]:https://pypi.org/project/meetup-api/
[3]:https://github.com/narrow-street/plan-your-meetup/blob/master/app/EDA.py
[4]:https://narrow-street.github.io/plan-your-meetup/docs/occurrence.html
[5]:https://github.com/narrow-street/plan-your-meetup/blob/master/app/viz_EDA.py
[6]:https://github.com/narrow-street/plan-your-meetup/blob/master/app/sklearn_models.py
[7]:https://github.com/narrow-street/plan-your-meetup/blob/master/app/keras_models.py
[8]:https://github.com/narrow-street/plan-your-meetup/blob/master/app/topic_classifier.py
[9]:https://narrow-street.github.io/plan-your-meetup/docs/classification.html
[10]:https://github.com/narrow-street/plan-your-meetup/blob/master/app/viz_classification.py
[11]: https://narrow-street.github.io/plan-your-meetup/docs/venues.html
[12]:https://github.com/narrow-street/plan-your-meetup/blob/master/app/spark_data.py
