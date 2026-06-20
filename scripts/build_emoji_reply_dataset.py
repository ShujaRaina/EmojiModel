"""Generate a synthetic *emoji-reply* dataset for instruction tuning.

Each row pairs a natural-language message with an emoji-only reply, grouped by
topic. The output is Alpaca-style instruction-tuning JSONL:

    {"instruction": "<message>", "input": "", "output": "<emoji reply>",
     "topic": "<topic>"}

For the text -> emoji diffusion pipeline, map `instruction` -> text and
`output` -> emoji.

Usage:
    python scripts/build_emoji_reply_dataset.py \
        --out data/emoji_reply/emoji_reply.jsonl --per-message 6 --seed 42
"""
import argparse
import itertools
import json
import os
import random

# topic -> (messages, emoji_pool). Emoji pools are intentionally a bit larger
# than any single reply so we can sample varied, on-topic combinations.
TOPICS = {
  'love': (
    ["I love you so much", "You mean everything to me", "I can't stop thinking about you",
     "Will you be my valentine", "You make me so happy", "I miss you already",
     "You're the love of my life", "My heart belongs to you", "I adore you",
     "Thinking of you all day", "You're my everything", "Forever yours",
     "I'm so lucky to have you", "You stole my heart", "Date night tonight"],
    ["❤️", "😍", "🥰", "💕", "😘", "💖", "💘", "💝", "😻", "💞", "🌹", "💋"]),
  'celebration': (
    ["I just got promoted", "We won the game", "It's my birthday today",
     "I passed my exam", "We're getting married", "I got the job",
     "Happy new year", "We did it", "Time to celebrate", "I graduated",
     "Best day ever", "Our team won the championship", "I got accepted",
     "Cheers to us", "Let's party tonight"],
    ["🎉", "🥳", "🎊", "👏", "🍾", "🥂", "🙌", "✨", "🎈", "🎁", "🏆", "🤩"]),
  'food': (
    ["Let's get pizza", "I'm so hungry", "This burger is amazing",
     "Time for dinner", "I made pasta tonight", "Who wants tacos",
     "Dessert time", "This is delicious", "I could eat all day",
     "Let's order sushi", "Breakfast is ready", "Craving some ice cream",
     "Movie night snacks", "Fresh coffee and donuts", "Barbecue this weekend"],
    ["🍕", "😋", "🤤", "🍔", "🍟", "🌮", "🍣", "🍦", "🍩", "🍰", "🍝", "🍗"]),
  'sad': (
    ["I'm feeling really down", "I miss you", "Today was rough",
     "I'm so heartbroken", "Everything is going wrong", "I feel so alone",
     "I just want to cry", "I'm exhausted and sad", "Nothing feels right",
     "I lost someone dear", "I'm so disappointed", "It hurts so much",
     "I had a terrible day", "I feel empty", "Why does it hurt"],
    ["😢", "💔", "🥺", "😞", "😔", "😭", "🫂", "😟", "☹️", "😿", "🌧️", "💧"]),
  'gym': (
    ["Going to the gym to work out", "Time to lift some weights", "Leg day today",
     "I just ran five miles", "Beast mode activated", "Crushing my workout",
     "No pain no gain", "Cardio time", "Hitting a new personal record",
     "Morning workout done", "Let's get strong", "Training hard today",
     "Sweat now shine later", "Pumping iron", "Marathon training"],
    ["💪", "🏋️", "🏃", "🔥", "🥵", "⚡", "🏅", "🤸", "🚴", "🧗", "😤", "🦵"]),
  'weather_sun': (
    ["It's so sunny today", "Beach day", "Perfect weather outside",
     "Summer vibes", "Let's go to the pool", "What a beautiful morning",
     "Clear blue skies", "Time to get a tan", "Heatwave this week",
     "Picnic in the park", "Sunshine and good times", "Out for a sunny walk",
     "Golden hour is gorgeous", "Sunset at the shore", "Warm and bright today"],
    ["☀️", "🏖️", "😎", "🌊", "🌞", "🏝️", "🌅", "🩴", "🍹", "🌴", "🕶️", "🌻"]),
  'weather_cold': (
    ["It's snowing outside", "So cold today", "Winter is here",
     "Let's build a snowman", "Bundle up it's freezing", "First snow of the year",
     "Hot cocoa weather", "Frosty morning", "Snow day", "Skiing this weekend",
     "Brr it's chilly", "Cozy by the fire", "Icy roads today", "Sweater season",
     "Snowball fight"],
    ["❄️", "⛄", "🥶", "🧣", "🧤", "☃️", "🔥", "🌨️", "⛷️", "🏂", "🧥", "🍫"]),
  'study': (
    ["I have exams next week", "So much homework", "Time to study hard",
     "Late night studying", "Final exams are coming", "I need to focus",
     "Library all day", "Cramming for the test", "Writing my thesis",
     "Group study session", "Reading for class", "Pulling an all nighter",
     "Math homework is tough", "Studying for finals", "Need to ace this exam"],
    ["📚", "✏️", "📖", "💻", "🤓", "😩", "📝", "🧠", "⏰", "☕", "📒", "🖊️"]),
  'work': (
    ["So much work to do", "Big meeting today", "Deadline is tomorrow",
     "Working from home", "Another busy day at the office", "Got a new project",
     "Email overload", "Presentation went well", "Overtime again",
     "Closing the big deal", "Back to back meetings", "Crunching the numbers",
     "Launching our product", "Team standup time", "Finishing the report"],
    ["💼", "💻", "📈", "📊", "🗓️", "⏳", "📞", "🤝", "📧", "🖥️", "😮‍💨", "✅"]),
  'sleep': (
    ["I'm exhausted", "Need some sleep", "So tired today", "Time for bed",
     "Can't keep my eyes open", "Goodnight everyone", "Nap time", "I'm so sleepy",
     "Up all night", "Dreaming already", "Bedtime", "Hit the snooze again",
     "Counting sheep", "Cozy under the blanket", "Early night tonight"],
    ["😴", "💤", "🥱", "🛌", "🌙", "🌜", "🛏️", "😪", "🦉", "⭐", "🌌", "🥴"]),
  'angry': (
    ["I'm so mad right now", "This is so frustrating", "I can't believe this",
     "That made me furious", "I'm fed up", "This is ridiculous",
     "I've had enough", "So annoyed", "This makes my blood boil",
     "Why does this keep happening", "I'm livid", "Absolutely unacceptable",
     "I'm raging", "This is infuriating", "Leave me alone"],
    ["😡", "🤬", "😤", "😠", "💢", "🔥", "👿", "🙄", "😾", "🗯️", "‼️", "💥"]),
  'funny': (
    ["That's hilarious", "I can't stop laughing", "lol so funny",
     "Best joke ever", "This is comedy gold", "I'm dying laughing",
     "You're so funny", "That cracked me up", "Hahaha good one",
     "My sides hurt", "Too funny", "Rolling on the floor", "What a meme",
     "Can't even", "This made my day"],
    ["😂", "🤣", "😆", "😹", "😜", "🙃", "😝", "🤪", "😄", "💀", "🤭", "😅"]),
  'travel': (
    ["Going on vacation", "Flying out tomorrow", "Road trip time",
     "Exploring a new city", "Packing my bags", "Adventure awaits",
     "Off to the mountains", "Backpacking through Europe", "Beach getaway booked",
     "Catching a flight", "Wanderlust calling", "Sightseeing all day",
     "New country new memories", "Cruise vacation", "Camping under the stars"],
    ["✈️", "🌍", "🧳", "🏔️", "🗺️", "🏝️", "🚗", "🛫", "🎒", "🏕️", "🚢", "📸"]),
  'animals': (
    ["I love my dog", "Look at this cat", "Puppies are the best",
     "My cat is so cute", "Saw a cute bunny", "Walking the dog",
     "Adopted a kitten", "Birds are singing", "Petting zoo today",
     "My pet is adorable", "Cuddling with my cat", "Doggo at the park",
     "Such a good boy", "Fluffy little friend", "Animal lover forever"],
    ["🐶", "🐱", "🐾", "🥰", "🐕", "🐈", "🐰", "🐹", "🦜", "🐢", "🐥", "❤️"]),
  'music': (
    ["Let's dance", "I love this song", "Turn up the music",
     "Concert tonight", "New album just dropped", "Singing in the shower",
     "This beat is fire", "Vinyl record collection", "Karaoke night",
     "Headphones on world off", "Live band rocks", "Music is my therapy",
     "Dancing all night", "DJ set was amazing", "Got the rhythm"],
    ["🎶", "🎵", "💃", "🕺", "🎧", "🎤", "🎸", "🥁", "🔊", "🎹", "🎷", "🎺"]),
  'coffee': (
    ["I need coffee", "Good morning everyone", "Time for my latte",
     "Coffee first then talk", "Fresh espresso please", "Morning fuel",
     "Cappuccino o'clock", "Can't function without caffeine", "Brewing a fresh cup",
     "Coffee and a good book", "Iced coffee weather", "One more cup",
     "Rise and grind", "Cozy cafe morning", "Decaf for me tonight"],
    ["☕", "🌅", "😌", "🥱", "💛", "🧋", "🍵", "😋", "🫖", "📖", "🌞", "✨"]),
  'congrats': (
    ["Congratulations", "Well done", "You did amazing", "So proud of you",
     "Great job", "You earned it", "Way to go", "Kudos to you",
     "What an achievement", "You nailed it", "Bravo", "Outstanding work",
     "Hats off to you", "You're a star", "Keep shining"],
    ["🎓", "👏", "🙌", "🌟", "🏆", "🥇", "💯", "🎉", "✨", "👍", "🤩", "💪"]),
  'scared': (
    ["That was so scary", "I'm terrified", "Did you hear that noise",
     "This place is creepy", "I have goosebumps", "Horror movie night",
     "Something spooked me", "I'm shaking", "That gave me chills",
     "Haunted house tonight", "So frightening", "My heart is racing",
     "Scared of the dark", "That jump scare got me", "Eerie vibes"],
    ["😱", "😨", "😰", "👻", "🙀", "😖", "🫣", "💀", "🕷️", "🦇", "🌑", "😳"]),
  'grateful': (
    ["Thank you so much", "I really appreciate it", "You're a lifesaver",
     "I'm so grateful", "Thanks a million", "Couldn't have done it without you",
     "You're too kind", "Much appreciated", "I owe you one",
     "Forever thankful", "You made my day", "So blessed", "Grateful for you",
     "Thanks for everything", "You're the best"],
    ["🙏", "😊", "💛", "🤗", "✨", "🥹", "💐", "🌷", "👍", "💯", "🫶", "😇"]),
  'excited': (
    ["I can't wait", "So excited for this", "This is going to be epic",
     "Counting down the days", "Best news ever", "I'm pumped",
     "Let's gooo", "So hyped right now", "This is amazing news",
     "Over the moon", "Thrilled beyond words", "Can't contain my excitement",
     "Big things coming", "Here we go", "Buzzing with energy"],
    ["🤩", "🎉", "🙌", "🔥", "✨", "😆", "🥳", "💥", "⚡", "😻", "🚀", "🌟"]),
  'sick': (
    ["I'm not feeling well", "I have a cold", "Caught the flu",
     "My head hurts", "Feeling under the weather", "Stuck in bed sick",
     "Sore throat today", "Need some rest", "Running a fever",
     "Achy all over", "Sneezing nonstop", "Tummy ache", "Get well soon to me",
     "Drinking tea to feel better", "Doctor's appointment today"],
    ["🤒", "🤧", "🤕", "🥴", "😷", "🛌", "🤢", "💊", "🌡️", "🍵", "🛏️", "😣"]),
  'morning': (
    ["Good morning", "Rise and shine", "A brand new day", "Up early today",
     "Morning everyone", "Fresh start today", "Sunrise jog", "Early bird gets the worm",
     "Stretching to start the day", "Breakfast then go", "Hello world",
     "Bright and early", "Let's make today great", "Morning motivation",
     "First light is beautiful"],
    ["🌅", "☀️", "😊", "☕", "🐦", "🌞", "💪", "✨", "🙂", "🌻", "🥞", "🌄"]),
  'night': (
    ["Good night", "Sweet dreams", "Stars are out", "Late night thoughts",
     "Time to wind down", "The moon is beautiful", "Quiet night in",
     "City lights at night", "Peaceful evening", "Under the night sky",
     "Night owl mode", "Calm and quiet", "Cozy night", "Dreamy evening",
     "See you tomorrow"],
    ["🌙", "🌛", "⭐", "🌌", "😴", "💤", "🌃", "✨", "🦉", "🛌", "🌠", "🕯️"]),
  'party': (
    ["Let's party", "Night out with friends", "Dance floor is calling",
     "Time to have fun", "Weekend vibes", "Club night", "Throwing a party",
     "Celebrate good times", "Drinks are on me", "Music's loud let's go",
     "Festival weekend", "Dancing till dawn", "Birthday bash tonight",
     "Good music good friends", "Turn up"],
    ["🎉", "🥳", "🪩", "💃", "🕺", "🍾", "🍻", "🎊", "🔊", "🎈", "✨", "🎶"]),
  'nature': (
    ["What a beautiful sunset", "Hiking in the forest", "Love being outdoors",
     "Flowers are blooming", "Sitting by the lake", "Fresh mountain air",
     "Watching the waves", "Stargazing tonight", "Autumn leaves are falling",
     "Garden is thriving", "Walk in the woods", "Rainbow after the rain",
     "Birdsong this morning", "Camping by the river", "Nature heals"],
    ["🌅", "🌲", "🌸", "🏞️", "🍃", "🌊", "⛰️", "🌈", "🍂", "🌻", "🦋", "🌿"]),
  'sports': (
    ["Game day", "Goooal", "Let's win this", "Cheering for my team",
     "Touchdown", "Basketball tonight", "What a match", "Home run",
     "Final whistle", "Tennis practice", "We're champions", "Race day",
     "Slam dunk", "Soccer with friends", "Go team go"],
    ["⚽", "🏀", "🏈", "⚾", "🎾", "🏆", "🥅", "🙌", "🔥", "🏅", "📣", "🎽"]),
  'rain': (
    ["It's raining heavily", "Rainy day vibes", "Don't forget your umbrella",
     "Thunderstorm tonight", "Love the sound of rain", "Puddles everywhere",
     "Stay dry out there", "Cozy rainy afternoon", "Storm rolling in",
     "Dancing in the rain", "Grey skies today", "Rain on the window",
     "Lightning in the distance", "Wet weather warning", "Petrichor smells great"],
    ["🌧️", "☔", "⛈️", "🌩️", "💧", "🌫️", "🌂", "⚡", "🫧", "🌦️", "🏠", "☁️"]),
  'tech': (
    ["My new phone arrived", "Coding all night", "The app finally works",
     "Love this gadget", "Upgrading my setup", "Bug squashed",
     "New laptop day", "Robots are the future", "Gaming session tonight",
     "Shipped the new feature", "Wifi is down again", "AI is amazing",
     "Building something cool", "Debugging marathon", "Tech support please"],
    ["💻", "📱", "🤖", "🚀", "🕹️", "⌨️", "🖥️", "🔌", "💡", "🛠️", "👾", "✨"]),
}

# light prefixes to add natural variety to messages without changing meaning
PREFIXES = ["", "", "", "omg ", "hey ", "honestly ", "ugh ", "wow ", "guys ", "ok "]


def emoji_combos(pool, k_min, k_max, count, rng):
  """Return up to `count` distinct ordered emoji-reply strings from `pool`."""
  seen, out, tries = set(), [], 0
  pool = list(pool)
  max_tries = count * 40
  while len(out) < count and tries < max_tries:
    tries += 1
    k = rng.randint(k_min, k_max)
    k = min(k, len(pool))
    combo = rng.sample(pool, k)
    key = ''.join(combo)
    if key in seen:
      continue
    seen.add(key)
    out.append(''.join(combo))
  return out


def build(per_message, k_min, k_max, seed):
  rng = random.Random(seed)
  rows, seen_pairs = [], set()
  for topic, (messages, pool) in TOPICS.items():
    for msg in messages:
      prefix = rng.choice(PREFIXES)
      body = msg
      # Lowercase the message's leading letter when prefixed, so it reads
      # naturally ("Ok concert tonight"), but keep the pronoun "I".
      if prefix and body.split()[0] not in ("I", "I'm", "I've", "I'll", "I'd"):
        body = body[0].lower() + body[1:]
      instruction = (prefix + body).strip()
      instruction = instruction[0].upper() + instruction[1:]
      for reply in emoji_combos(pool, k_min, k_max, per_message, rng):
        pair = (instruction, reply)
        if pair in seen_pairs:
          continue
        seen_pairs.add(pair)
        rows.append({'instruction': instruction, 'input': '',
                     'output': reply, 'topic': topic})
  rng.shuffle(rows)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default='data/emoji_reply/emoji_reply.jsonl')
  ap.add_argument('--per-message', type=int, default=6)
  ap.add_argument('--k-min', type=int, default=2)
  ap.add_argument('--k-max', type=int, default=5)
  ap.add_argument('--seed', type=int, default=42)
  args = ap.parse_args()

  rows = build(args.per_message, args.k_min, args.k_max, args.seed)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  with open(args.out, 'w', encoding='utf-8') as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + '\n')

  topics = sorted({r['topic'] for r in rows})
  print(f'Wrote {len(rows)} rows across {len(topics)} topics -> {args.out}')
  print('Sample rows:')
  for r in rows[:8]:
    print(f"  {r['instruction']!r} -> {r['output']}  ({r['topic']})")


if __name__ == '__main__':
  main()
