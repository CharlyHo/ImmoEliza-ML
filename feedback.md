# Feedback
- main branch is the only branch I normally look at
- I don't understand well the git tree with the commits
- not meaningful commit messages "file name", "modif file name", ... It has to be clearly understandable
- no readme! 
- weird folder name "data handled differently", use convention (with _ instead of space) and make a data folder with different version of the dataset if needed but the main one must be clearly named
- Xgboost_optuna.py should be splitted in more files to avoid having such a large file but the code is quite clean, missing dynamic typing
  - utils.py for cleaning or preprocessing
  - model.py for training the model
  - eval.py for model evaluation
  - ...
- I don't know what to do with Charly's model folder? Without a proper readme I tend to look only at the root of the folder because I don't have the key to understand if the other folders are there for archive purpose or if it's part of the project? A good usage of git would have been better with clear commits and merges, on this branch I don't have a good view on the team effort. You could have created 1 file per model in your own branch to investigate, pick the bests results, merge in main and then clean everything else.

This is almost a complete project, good job :fire:! The most important is the learning challenge you faced.

A trick you can use is to assign a repo-manager when you start a project, that person would be in charge of creating a clear repo. 

You did not put the metrics of your best model in the spreadsheet.

It's not easy to work as a team on a project but this is important to make the project as readable as possible.


## Evaluation criteria

| Criteria       | Indicator                                     | Yes/No |
| -------------- | --------------------------------------------- | ------ |
| 1. Is complete | Know how to answer all the above questions.   | YES    |
|                | `pandas` and `matplotlib`/`seaborn` are used. | YES    |
|                | All the above steps were followed.            | YES    |
|                | A nice README is available.                   | NO    |
|                | Your model is able to predict something.      | YES    |
| 2. Is good     | You used typing and docstring.                | +/-    |
|                | Your code is formatted (PEP8 compliant).      | YES    |
|                | No unused file/code is present.               | NO    |
