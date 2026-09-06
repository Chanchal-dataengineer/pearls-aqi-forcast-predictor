# The Story Behind “Pearls AQI Forecast Predictor”

When I started working on this project for my 10Pearls submission, I did not want to make just another basic machine learning project. I wanted to work on something that was connected to a real problem and also had some meaning for me.

That is how I came up with the idea of building an air quality prediction system for **Nagarparkar, Sindh**.

My main goal was simple: **Can I use available online data and machine learning to monitor and predict air quality for a place where proper air quality monitoring stations are not easily available?**

---

## Why Did I Choose Nagarparkar?

When we talk about environmental problems in Pakistan, most projects and discussions focus on big cities like Karachi, Lahore, and Islamabad.

But there are many smaller and remote areas that also deserve attention.

I chose **Nagarparkar, Sindh**, because it is a beautiful and historic area near the Karoonjhar Mountains. It has its own culture, natural beauty, tourism, and local communities. At the same time, like many remote areas, it does not have the same level of access to expensive air quality monitoring systems and physical monitoring stations as major cities.

This made me think about a simple question:

**If a place does not have expensive monitoring hardware, does that mean we cannot monitor its air quality?**

I wanted to find an answer to that question through this project.

My idea was to use publicly available weather and air pollution data through the **OpenWeather API**, process that data, create useful features, and then use it for air quality prediction.

So, instead of depending completely on physical hardware, I wanted to see how far we could go using **data, Python, machine learning, and cloud-based automation**.

---

## What Was My Actual Target?

My target was not only to train a machine learning model and show an accuracy score.

I wanted to build a **complete working system**.

The idea was to make a pipeline that could:

* Collect air quality and weather-related data automatically.
* Get important pollution values such as **PM2.5, PM10, CO, NO, NO2, O3, SO2, and NH3**.
* Clean and prepare the collected data.
* Create useful time-based features such as hour, day, month, and weekday.
* Calculate changes in AQI and rolling averages to understand recent trends.
* Use the prepared data for prediction.
* Automate the process so that the system does not depend on me running everything manually.
* Provide a simple way for a user to understand the predicted air quality.

So, my focus was on building something closer to a **real-world end-to-end ML system**, rather than keeping everything inside one Jupyter Notebook.

---

## How I Built the Project

I divided the project into different stages so that every step had a clear purpose.

First, I collected air quality data for **Nagarparkar** using the OpenWeather API. The collected data included AQI and different pollutants such as PM2.5, PM10, CO, NO, NO2, O3, SO2, and NH3.

After collecting the data, I moved to the preprocessing stage. I checked the dataset for missing values, duplicate records, and data types. I also converted the datetime information into the proper datetime format so that I could work with the time-based information correctly.

After that, I worked on **feature engineering**.

I created features such as:

* Hour
* Day
* Month
* Weekday
* AQI Change
* AQI Rolling Average

These features helped me look at the data not only as individual values, but also as a time-based pattern.

For example, the rolling average helped me understand the recent AQI trend instead of looking at only one reading.

---

## More Than Just a Jupyter Notebook

One thing I really wanted to learn through this project was that a machine learning project should not end after training a model.

In many university projects, we train a model in a notebook, get an accuracy score, and then consider the project complete.

But I wanted to understand what happens when we try to make the project work continuously.

That is why I worked toward building an automated pipeline.

The project includes:

* **Data collection** through the OpenWeather API.
* **Data preprocessing** to clean and prepare the information.
* **Feature engineering** to create useful prediction features.
* **Machine learning models** such as Random Forest and Ridge as part of the prediction framework.
* **Automation through GitHub Actions**, so the workflow can run without manually pressing “Run” every time.
* A **user-friendly prediction/dashboard side** so the final result can be understood more easily.

This was one of the most important parts of the project for me because it helped me understand the difference between simply building a model and building a complete ML system.

---

## What I Learned from the Data

The most interesting part of this project was that the real data did not behave exactly like the datasets we normally see in tutorials.

When I collected and analyzed the Nagarparkar data, I noticed that the air quality was mostly very clean and stable during my tracking period. The AQI value stayed around **1 most of the time**.

At first, this was a little surprising because I expected the machine learning model to show more variation.

But then I realized that this was actually an important finding.

The data was telling me something about the actual location.

Because the AQI did not change much during the period I was monitoring, even a simple approach based on the current AQI being similar to the next hour could perform very well.

This taught me an important lesson:

**A high prediction score does not always mean that the model is highly intelligent. Sometimes, the nature of the data itself makes the prediction easier.**

I did not want to hide this fact or change the data just to make my machine learning model look better.

Instead, I decided to keep the results honest and use the project as a learning and engineering framework.

---

## Why I Still Kept the Machine Learning Models

Even though the data was quite stable during my tracking period, I kept the machine learning part of the project.

I used **Random Forest and Ridge** as part of the experimental prediction framework.

The reason was simple: the current data represents only the conditions observed during my collection period. Air quality can change with weather, seasons, wind, dust, traffic, human activity, and other environmental factors.

If the system gets more varied data in the future, the machine learning models can become more useful.

So, I did not build the project only for the data I had at that moment.

I wanted to build a system that has the basic structure needed to work with changing data in the future.

---

## What This Project Means to Me

For me, **Pearls AQI Forecast Predictor** is more than just a Python or machine learning project.

It helped me understand how a real project moves from **raw data → preprocessing → feature engineering → machine learning → automation → final prediction**.

It also taught me that real-world data is not always perfect or exciting. Sometimes the most important result is simply what the data tells you.

Most importantly, this project gave me the opportunity to work on a problem related to a place that is usually not the main focus of large environmental projects.

My goal was to show that even without expensive physical monitoring hardware, we can still explore useful solutions using **open data, APIs, Python, machine learning, and automation**.

This project is my step toward understanding how data science and machine learning can be used for real problems—not just for getting a good accuracy score, but for building something that can actually be improved and used in the future.

**Thank you for reading about my project and the story behind it.**
