# BackBuddy

![BackBuddy](https://github.com/IdeaKing/BlairHacks/blob/main/docs/BackBuddyLogo.png)

## Examples

![BackBuddy Example 1](https://github.com/IdeaKing/BlairHacks/blob/main/docs/Figure_1.png)

![BackBuddy Example 2](https://github.com/IdeaKing/BlairHacks/blob/main/docs/Figure_2.png)

## Inspiration

## What it does
BackBuddy is a simple GUI application that takes an input video from a user, and passes it to an AI algorithm. This algorithm is a keypoint detector, which tracks the joints of a person. Then, we use this keypoint model to analyze the positions of the hands and feet on a person performing a deadlift, and use geometric calculations to evaluate the accuracy of the lift.
## How we built it
First, we collected data for our keypoints algorithm, and for our final deadlift calculations. We did this by web scraping YouTube videos, using Supervisely to provide a label for each video, and then training a TensorFlow-MobileNetSlim model. While the model was training, we prepared our data further by splitting each video into separate frames, and augmented the data 4 times to create 17,694 individual JPG files in total. After this was done, we pulled keypoints from each of the videos, and created a function to calculate accuracy based on the foot keypoint coordinates and the feet keypoint coordinates.
## Challenges we ran into
At first, we attempted to create and design our own keypoints algorithm from scratch, but were unable to do so successfully with good accuracy. We also attempted to use logistic regression to analyze the keypoints of each video, with the labels that we had already created. However, different videos contained different amounts of keypoints, as some joints were not visible. This led to difficult data preprocessing, so we were unable to use our ML algorithm to classify inputs.
## Accomplishments that we're proud of


## What we learned
A few things we learned while creating our project was that 
## What's next for BackBuddy
For future goals, we want to implement live camera streaming to judge their level of deadlift. We have attempted to add a live camera, but due to the time restraint we were not able to successfully work it into our program. We also want to add logistic regression to rate the deadlift for a more accurate classification system to better the users deadlift to ensure the greatest level of safety for the athlete.


