# ezdata

A custom Python library intended to help with processing and visualizing survey data.

## Status: Work in progress
This is a very early work in progress. I'm currently working on basic pre-processing and cleaning tools before building out the plotting capabilities. I'm intending to build upon Plotly for the visualization.

This is also intended primarily for my personal use. As such, it will be very highly opinionated and likely make many assumptions about what a prospective user (i.e., yours truly) would like. :)

## Why
In order to create data visualizations to share with others, I've found myself repeatedly doing the same pre-processing or writing very similar expressions to create graphs (e.g., sorting, filtering, etc.).

While the availability of many high-quality packages for a variety of tasks has been great, it also can be clunky when I'm just trying to get something done. Remembering which package I need for what task and the differences in the APIs can be (minor) barriers to realizing my intent.

Years ago, I started what I'll admit to be a very messy and monolithic "figs.py" file containing custom classes that I used to simplify graph creation. While it works, it was in desperate need of refactoring. It also was scoped specifically to creating visualizations and very limited in its capabilities (e.g., only included standard Z or Wald confidence intervals).

In addition to that, I wanted to take this opportunity to build good (or better?) coding habits and develop familiarity with version control and Git. 
 