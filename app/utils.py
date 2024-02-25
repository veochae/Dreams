from better_profanity import profanity

def task(index , xx):
    return(index,profanity.censor(xx, "*"))