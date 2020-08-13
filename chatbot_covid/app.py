# import important packages
from flask import Flask,jsonify,request,render_template
import flair # sentiment detection models
from bert_embedding import BertEmbedding # embedding
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist # similarity models

# define an app
app = Flask(__name__)

# define function for sentiment detection
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

# sentiment analysis for a sentence
def sentiment(text,flair_sentiment):
    '''
    function that uses flair off-the-shelf sentiment analysis model
    text for which sentiment has to be detected
    flair_sentiment is sentiment analysis model 
    '''
    s = flair.data.Sentence(text)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    if total_sentiment[0].value == 'POSITIVE' and total_sentiment[0].score > 0.9:
        return True
    else:
        return False

# define function for calculating similarity
bm = BertEmbedding()
model = SentenceTransformer('bert-base-nli-mean-tokens')

# function that finds similarity to yes
def sim_yes(text,model):
    
    text = text.strip()
    if not(text.endswith('.')):
        text = text + "."
    corpus = ['yes.','no.',text]
    text_split = [t.lower() for t in text.strip('.').split()]
    if 'yes' in text_split:
        return True
    elif 'no' in text_split:
        return False
    else:
        corpus_embeddings = model.encode(corpus)
        distance1 = cdist([corpus_embeddings[0]],[corpus_embeddings[2]],"cosine")
        distance2 = cdist([corpus_embeddings[1]],[corpus_embeddings[2]],"cosine")
        return (distance1 < distance2)[0][0]

# function that finds similarity to I am done
def sim_done(text,model):
    
    text = text.strip()
    if not(text.endswith('.')):
        text = text + "."
    corpus = ['I am Done.','I am not Done',text]
    corpus_embeddings = model.encode(corpus)
    distance1 = cdist([corpus_embeddings[0]],[corpus_embeddings[2]],"cosine")
    distance2 = cdist([corpus_embeddings[1]],[corpus_embeddings[2]],"cosine")
    return (distance1 < distance2)[0][0]

# defining the dictionary that will be exchanged with mobile app
xinfo = {
    'text': 'Hi, I am Betty. So, how are you feeling today?',
    'status': 'active',
    'block': 'greeting',
    'section': 1,
    'action': 'talk'
}

# post messages to server
@app.route('/voice/greeting/' , methods=['POST'])
def mobile2server_greeting():

	global xinfo
	# requesting data from mobile
	request_data = request.get_json()

	# block to be processed greeting
	if request_data['block'] == 'greeting' and request_data['section'] == 1:

		xinfo = {
		    'text': '',
		    'status': 'active',
		    'block': 'greeting',
		    'section': 1,
		    'action': 'talk'
		}

		# if user is feeling good
		if sentiment(request_data['text'],flair_sentiment):
			xinfo['text'] = 'Great, I don’t think you would need our self-assessment tool today. Am I right?'
			xinfo['section'] = 2
			return jsonify(xinfo)

		else:
			xinfo['text'] = 'Hmm, are you experiencing any one of these problems? Severe chest pain, severe difficulty breathing, losing consciousness, feeling confused/disoriented?'
			xinfo['section'] = 3
			return jsonify(xinfo)

	elif request_data['block'] == 'greeting' and request_data['section'] == 2:

		xinfo = {
		    'text': '',
		    'status': 'active',
		    'block': 'greeting',
		    'section': 2,
		    'action': 'talk'
		}

		# assurance that person is indeed feeling good
		if sim_yes(request_data['text'],model):
		    xinfo['text'] = 'Thanks for chatting with me. Take care and be safe.'
		    xinfo['action'] = 'talk'
		    xinfo['block'] = 'exit'
		    xinfo['section'] = 0
		    return jsonify(xinfo)

		else:
			xinfo['text'] = 'Hmm, are you experiencing any one of these problems? Severe chest pain, severe difficulty breathing, losing consciousness, feeling confused/disoriented?'
			xinfo['section'] = 3
			return jsonify(xinfo)

	elif request_data['block'] == 'greeting' and request_data['section'] == 3:

		xinfo = {
		    'text': '',
		    'status': 'active',
		    'block': 'greeting',
		    'section': 3,
		    'action': 'talk'
		}

		# assurance that person is indeed feeling good
		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'You need immediate help. Let me call nine-one-one.'
			xinfo['block'] = 'exit'
			xinfo['section'] = 0
			xinfo['action'] = 'call_911'
			return jsonify(xinfo)

		else:
			xinfo['text'] = 'Are you currently experiencing fever or chills?'
			xinfo['section'] = 1
			xinfo['block'] = 'syn_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)
	else:
		return jsonify ({'error': 'Required data not present.'})

# get messages from server
@app.route('/voice/greeting/')
def server2mobile_greeting():

	if xinfo['block'] == 'greeting':
		return jsonify(xinfo)

	else:
		return jsonify ({'error': 'server not in greeting block'})

# post messages to server
@app.route('/voice/syn_covid/' , methods=['POST'])
def mobile2server_syncovid():

	global xinfo
	# requesting data from mobile
	request_data = request.get_json()

	# block to be processed when asking for symptom 1
	if request_data['block'] == 'syn_covid' and request_data['section'] == 1:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['section'] = 1
			xinfo['action'] = 'display_centers'
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Is your cough new or worsening or making whistling noise?'
			xinfo['section'] = 2
			xinfo['block'] = 'syn_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 2
	elif request_data['block'] == 'syn_covid' and request_data['section'] == 2:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['section'] = 1
			xinfo['action'] = 'display_centers'
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Are you experiencing shortness of breath or difficulty in breathing?'
			xinfo['section'] = 3
			xinfo['block'] = 'syn_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 3
	elif request_data['block'] == 'syn_covid' and request_data['section'] == 3:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['action'] = 'display_centers'
			xinfo['section'] = 1
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Is your throat sore or do you have difficulty in swallowing?'
			xinfo['section'] = 4
			xinfo['block'] = 'syn_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 4
	elif request_data['block'] == 'syn_covid' and request_data['section'] == 4:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['action'] = 'display_centers'
			xinfo['section'] = 1
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Is your nose stuffy, runny, congested or have loss of sense of smell or taste?'
			xinfo['section'] = 5
			xinfo['block'] = 'syn_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 5
	elif request_data['block'] == 'syn_covid' and request_data['section'] == 5:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['action'] = 'display_centers'
			xinfo['section'] = 1
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Do you have a pink eye or a headache that is long lasting or unusual?'
			xinfo['section'] = 6
			xinfo['block'] = 'syn_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 6
	elif request_data['block'] == 'syn_covid' and request_data['section'] == 6:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['action'] = 'display_centers'
			xinfo['section'] = 1
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Are you having digestive issues such as nausea, vomiting, diarrhea, stomach pain?'
			xinfo['section'] = 7
			xinfo['block'] = 'syn_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 7
	elif request_data['block'] == 'syn_covid' and request_data['section'] == 7:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['action'] = 'display_centers'
			xinfo['section'] = 1
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Are you having muscle ache or extreme tiredness that is unusual or do you feel like falling down?'
			xinfo['section'] = 8
			xinfo['block'] = 'syn_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 8 
	elif request_data['block'] == 'syn_covid' and request_data['section'] == 8:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['action'] = 'display_centers'
			xinfo['section'] = 1
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Hmm, I would like to know if you are 70 years or older.'
			xinfo['block'] = 'covid_prone'
			xinfo['action'] = 'talk'
			xinfo['section'] = 1
			return jsonify(xinfo)

	else:
		return jsonify ({'error': 'Required data not present.'})

# get messages from server
@app.route('/voice/syn_covid/')
def server2mobile_syncovid():

	if xinfo['block'] == 'syn_covid':
		return jsonify(xinfo)

	else:
		return jsonify ({'error': 'server not in syn covid block'})


# post messages to server
@app.route('/voice/display_centers/' , methods=['POST'])
def mobile2server_displaycenters():

	global xinfo
	# requesting data from mobile
	request_data = request.get_json()

	if request_data['block'] == 'display_centers' and request_data['section'] == 1:
		if sim_done(request_data['text'],model):
			xinfo['text'] = 'Stay at home and monitor your health. Here are some of the ways of doing that.'
			xinfo['block'] = 'exit'
			xinfo['section'] = 0
			xinfo['action'] = 'talk'
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Thank you for chatting with me. Stay at home and be safe.'
			xinfo['block'] = 'exit'
			xinfo['section'] = 0
			xinfo['action'] = 'talk'
			return jsonify(xinfo)
	else:
		return jsonify ({'error': 'Required data not present.'})

# get messages from server
@app.route('/voice/display_centers/')
def server2mobile_displaycenters():

	if xinfo['block'] == 'display_centers':
		return jsonify(xinfo)

	else:
		return jsonify ({'error': 'server not in display centers block'})


# post messages to server
@app.route('/voice/covid_prone/' , methods=['POST'])
def mobile2server_covidprone():

	global xinfo
	# requesting data from mobile
	request_data = request.get_json()

	# block to be processed when asking for symptom 1
	if request_data['block'] == 'covid_prone' and request_data['section'] == 1:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answer, you are in the high risk category. You must stay at home and take all precautions to avoid contact with people, especially the ones who are sick or might have travelled aboard in the past 14 days.'
			xinfo['block'] = 'exit'
			xinfo['section'] = 0
			xinfo['action'] = 'talk'
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Do you have a condition that compromises immune system such as Lupus, Rheumatoid arthritis?'
			xinfo['section'] = 2
			xinfo['block'] = 'covid_prone'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 2
	elif request_data['block'] == 'covid_prone' and request_data['section'] == 2:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answer, you are in the high risk category. You must stay at home and take all precautions to avoid contact with people, especially the ones who are sick or might have travelled aboard in the past 14 days.'
			xinfo['block'] = 'exit'
			xinfo['section'] = 0
			xinfo['action'] = 'talk'
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Do you have chronic health condition such as diabetes, asthma, emphysema?'
			xinfo['section'] = 3
			xinfo['block'] = 'covid_prone'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 3
	elif request_data['block'] == 'covid_prone' and request_data['section'] == 3:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answer, you are in the high risk category. You must stay at home and take all precautions to avoid contact with people, especially the ones who are sick or might have travelled aboard in the past 14 days.'
			xinfo['block'] = 'exit'
			xinfo['section'] = 0
			xinfo['action'] = 'talk'
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'Do you regularly go or have recently been to healthcare centre for services such as dialysis, cancer treatment, or surgery'
			xinfo['section'] = 4
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	# block to be processed when asking for symptom 4
	elif request_data['block'] == 'covid_prone' and request_data['section'] == 4:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answer, you are in the high risk category. You must stay at home and take all precautions to avoid contact with people, especially the ones who are sick or might have travelled aboard in the past 14 days.'
			xinfo['block'] = 'exit'
			xinfo['section'] = 0
			return jsonify(xinfo)
		else:
			xinfo['text'] = 'In the past 14 days, have you been in close physical contact with people who tested positive for COVID-19 or have been unusually sick, for more than 15 minutes?'
			xinfo['section'] = 1
			xinfo['block'] = 'expose_covid'
			xinfo['action'] = 'talk'
			return jsonify(xinfo)

	else:
		return jsonify ({'error': 'Required data not present.'})

# get messages from server
@app.route('/voice/covid_prone/')
def server2mobile_covidprone():

	if xinfo['block'] == 'covid_prone':
		return jsonify(xinfo)

	else:
		return jsonify ({'error': 'server not in covid prone block'})


# post messages to server
@app.route('/voice/expose_covid/' , methods=['POST'])
def mobile2server_exposecovid():

	global xinfo
	# requesting data from mobile
	request_data = request.get_json()

	# block to be processed when asking for symptom 1
	if request_data['block'] == 'expose_covid' and request_data['section'] == 1:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answers you must get a COVID-19 self-assessment test done at a centre closest to you, let me find one for you. Please say I am done, when you are done viewing the list of centres.'
			xinfo['block'] = 'display_centers'
			xinfo['action'] = 'display_centers'
			xinfo['section'] = 1

		else:
			xinfo['text'] = 'In the past 14 days, have you or any person living with you, travelled abroad or to the nearby province?'
			xinfo['block'] = 'travel_covid'
			xinfo['action'] = 'talk'
			xinfo['section'] = 1

	else:
		return jsonify ({'error': 'Required data not present.'})

# get messages from server
@app.route('/voice/expose_covid/')
def server2mobile_exposecovid():

	if xinfo['block'] == 'expose_covid':
		return jsonify(xinfo)

	else:
		return jsonify ({'error': 'server not in expose covid block'})

# post messages to server
@app.route('/voice/travel_covid/' , methods=['POST'])
def mobile2server_travelcovid():

	global xinfo
	# requesting data from mobile
	request_data = request.get_json()

	# block to be processed when asking for symptom 1
	if request_data['block'] == 'travel_covid' and request_data['section'] == 1:

		if sim_yes(request_data['text'],model):
			xinfo['text'] = 'Based on your answer you must self-isolate. Be sure to take this assessment test if you experience any unpleasantness. When, you are done please say I am done to proceed.'
			xinfo['block'] = 'exit'
			xinfo['action'] = 'talk'
			xinfo['section'] = 0

		else:
			xinfo['text'] = 'Great, based on your answers you don’t need an assessment for COVID-19. Avoid unnecessary travel and take all necessary precautions.'
			xinfo['block'] = 'exit'
			xinfo['action'] = 'talk'
			xinfo['section'] = 0

	else:
		return jsonify ({'error': 'Required data not present.'})

@app.route('/voice/travel_covid/')
def server2mobile_travelcovid():

	if xinfo['block'] == 'travel_covid':
		return jsonify(xinfo)

	else:
		return jsonify ({'error': 'server not in expose covid block'})

app.run(port=5000)
