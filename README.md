# BUIDL Bootcamp

If you're on Windows, [watch this video](https://youtu.be/qVQO7pezDWM). Otherwise:

### Step 1: Git

Git is the primary collaboration software for software development -- like Google Docs for programmers.

To check whether you have it installed go to your terminal, type “git” and hit enter. If you don’t see a bunch of instructions (see pic below), you don’t have git installed and need to install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

![image](./images/git.png)

### Step 2: Github

[Create a GitHub account](https://github.com/join) if you don’t already have one.

### Step 3: Python

Go to your terminal, type “python3” and hit enter. If you don’t see a “python prompt” (see pic below) then you need to [install Python 3](https://docs.python-guide.org/starting/installation/). Hit control-d or call the python `exit()` function to exit.

![image](./images/prompt.png)


### Step 4: Clone the Project with Git, install dependencies, and run it

```
git clone https://github.com/justinmoon/digital-cash.git
cd digital-cash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter notebook
```

This will open up your web browser to a page that looks like the image below. This is [Jupyter Notebook](http://jupyter.org/), which is basically a little website that’s run on only your computer.

Click the "PNGCoin Workbook.ipynb" link to follow along with this first lesson.
