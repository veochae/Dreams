from better_profanity import profanity

def task(index , xx):
    print("working")
    return(index,profanity.censor(xx, "*"))