## Introduction

In this project, we will be looking at building an `LLM from Scratch` and at various ways of accessing the LLMs. When it comes to building an LLM we would particularly look at the following pieces:

- Building LLM inputs
- Pretraining LLMs
- Fine tuning LLMs  

*Note:* 

1. All the requirements have been captured in requirements.txt. We have used `Poetry` for performing the package mangement.
2. Although we are following [Sebastian Raschka's tutorial](https://www.youtube.com/watch?v=quh7z1q7-uc) (a huge shout out to him for compiling this brilliant tutorial), we have made efforts to explore each concept he has so passionately mentioned in this tutorial. Loud the efforts of such great visionaries.  

## Various ways of using pre-trained LLMs

- Via Public & proprietary services

We have so dearly used the `OpenAI` models:

**Prompt**
```
Please come up with a dandy, attractive and creative description for an event based on following event title, event type, event extra information.
Title: "long separated brothers"
Type: Fraternity Union
Extra Information: Wheelchair accessible, Kid Friendly, Non Alcoholic, Family Friendly, No pets allowed.
```
**Chatgpt 4o Mini Response**
```
Event Title: Long Separated Brothers

Event Type: Fraternity Union

Event Description:

Step into a heartwarming chapter of reunion and camaraderie at Long Separated Brothers, an enchanting fraternity union that celebrates the rediscovery of cherished bonds. This unique event invites you to witness a joyful reconnection as brothers come together in a setting that blends elegance with inclusivity.

Held at a venue that is both wheelchair accessible and family-friendly, Long Separated Brothers ensures that every detail is designed to make the evening welcoming for all. Picture an atmosphere where vibrant decorations and cozy seating arrangements create a space that feels like a warm embrace, inviting guests of all ages to partake in the celebration.

Indulge in a feast of delectable, non-alcoholic delights and kid-friendly treats that cater to all tastes. The culinary offerings will be as diverse and delightful as the stories shared throughout the evening. Engage in interactive activities that bring out the best of both nostalgia and new memories, from a “Brotherly Bonding Game” to a “Memory Lane” photo gallery showcasing heartwarming moments from each brother’s journey.

With live music and entertainment designed to appeal to all generations, the event promises to be a joyful celebration of unity and fraternity. The evening will be filled with heartfelt speeches, fun games, and an array of activities that highlight the joy of rekindling old connections in a setting that is both inclusive and respectful.

Long Separated Brothers is more than a reunion; it’s a heartfelt celebration where every detail is crafted to honor the strength of brotherhood and the joy of coming together. Join us for a night of shared laughter, meaningful connections, and the pure delight of family unity, all in a setting designed with everyone’s comfort in mind.
```

-  Using Open Source Models

**Model in litgpt**

The following code was used:
```
>>> import torch
>>> device = torch.device("mps")
>>> from litgpt import LLM
>>> llm = LLM.load("microsoft/Phi-3-mini-4k-instruct")
>>> text = llm.generate("Please come up with a dandy, attractive and creative description for an event based on following event title, event type, event extra information. Title: long separated brothers, Type: Fraternity Union, Extra Information: Wheelchair accessible, Kid Friendly, Non Alcoholic, Family Friendly, No pets allowed.",  max_new_tokens=1024)
print(text)
```
**Note:** As of September 5, we got the following warning while using `litgpt` on Mac with M3 chip eventhough torch.mps.is_available() returned True value.

```
UserWarning: The operator 'aten::index_copy.out' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/mps/MPSFallback.mm:13.)
```

**Response**
```
Join us for an unforgettable evening dedicated to fraternity unity and brotherly love under the cheerful glow of the tifo screens, surrounded by the lively buzz of family merriment and warm gatherings. At our Grand Event Series, we're excited to host the delightful event titled "Brothers Reunite: A Welcoming Fraternity Union Night" that warmly celebrates the bonds of fraternity brotherhood with a special focus on long separated siblings.

This budget-friendly and child-friendly soirée promises an enchanting atmosphere where families and little ones can revel in the joyful essence of brotherhood. Our excellent accessibility features ensure an inclusive and smooth experience for friends from all walks of life, including those who navigate the world in wheelchairs. Come let us immerse in the vibrant spirit of unity, camaraderie, and love - with no alcohol in sight to cast away any concern of judgment or inconvenience.

When you step into our wonderful, wheelchair-friendly venue, bear in mind that we are a family-friendly ambiance, which means all our guests can indulge in the fun and colors without worry. From laughter-filled games organized by our skilled organizers to interactive activities that have parents engaging with their kids, we make sure every moment is a cherished memory making, shared with the closest ones.

While we understand the necessity to have a safe space, please note that pets are strictly prohibited at our Fraternity Union Night. This rule permits us to maintain a serene ambiance and protect our guests from any unexpected allergies or disturbances.

Whether you're a fulltime or an esteemed alumni, young or old, in the presence of your long-lost fraternity brothers or simply wish to experience the heartwarming fraternity spirit, we invite you to our Brothers Reunite Night. Here, we celebrate the bonds of brotherhood, promote unity, and reinforce the fraternity's ideals becoming ever more vivid and enjoyable for all who are eager to share it.

Come and revel in the strong fraternity ties that ignite joy, ignites the heart, and creates beautiful chapters in your life stories. To all our brothers reuniting after long-separated terms, together let's revel in our unique brotherhood that gains enrichment with each passing year. So dive into these cherished moments, reminisce about the fond memories, and let's prepare ourselves for the exciting journey ahead. 

Join us at "Brothers Reunite: A Welcoming Fraternity Union Night" – we warmly await your presence while promising corners of warm-hearted laughter, cheerful reunions, and countless joyous moments!
```
