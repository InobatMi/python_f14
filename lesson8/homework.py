
word_counts = {}
with open('homework.txt', 'r', encoding='utf-8') as file: 
    text = file.read()  
    words = text.lower().split() 
    for word in words:
            word = word.strip(".,!?-*")
            if word:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1


print(word_counts)