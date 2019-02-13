# Plan Your Meetup

## Summary: 

The goal of this project is to provide various insights about meetup planning and help hosts better organize their events. In particular, this project included the following sections:

1. Explorative analysis of [meetup occurrence][1] and [venue locations][2] ([source][3])
2. [Document classification][4] on meetup group descriptions  ([source][5])
3. Hybrid recommender system for event venues (ongoing)

## Dataset:
The **dataset** was fetched from the [Meetup API](https://www.meetup.com/meetup_api/) utilizing a python [API client](https://pypi.org/project/meetup-api/). In particular, We first identified the Meetup groups(>100 members) in the top 30 US cities with the most number of groups and then gathered the meetup details associated with these groups between 2011 and 2018 July. The raw dataset is around 10G.


[1]:https://narrow-street.github.io/plan-your-meetup/docs/occurrence.html
[2]:https://narrow-street.github.io/plan-your-meetup/docs/venues.html
[3]: https://github.com/narrow-street/plan-your-meetup/blob/master/app/explorative.py
[4]:https://narrow-street.github.io/plan-your-meetup/docs/classification.html
[5]:https://github.com/narrow-street/plan-your-meetup/blob/master/app/topic_classifier.py

