import pickle
import os
import w2v

with open(os.getcwd() + '/data/emotions_lexicon.p', 'rb') as f:
	data = pickle.load(f)

new_emotions_lexicon = {}
for k, v in data.items():
	sentiment = v['sentiment']
	emotions = v['emotion']
	if len(emotions) > 1:
		try:	
			topics = w2v.get_topic(k, emotions)		
			new_emotions_lexicon[k] = {'emotions':topics, 'sentiment':sentiment}
		except:
			print(k, 'word not in dictionary')
			continue

with open('emotions_lexicon_with_probs.p', 'wb') as handle: 
	pickle.dump(new_emotions_lexicon, handle, protocol=pickle.HIGHEST_PROTOCOL)


