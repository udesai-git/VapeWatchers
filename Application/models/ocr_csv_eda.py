import pandas as pd
import io
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
# Download required NLTK resources if you haven't already
nltk.download('punkt_tab')
nltk.download('stopwords')
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from collections import Counter
import textstat
import emoji


def tokenize_and_preprocess(df):
    """
    Tokenize the extracted text from OCR and add as a new column.

    Args:
        df: DataFrame containing 'extracted_text' column

    Returns:
        DataFrame with new 'tokens' column
    """
    # Create a function to process each text entry
    def process_text(text):
        if not isinstance(text, str) or not text.strip():
            return []

        # Convert to lowercase
        text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]


        return tokens

    # Apply the function to create a new column
    df['tokens'] = df['extracted_text'].apply(process_text)

    return df


def comprehensive_token_analysis(df):
    """
    Performs comprehensive analysis on tokenized text to detect youth-targeted marketing language.

    Args:
        df: DataFrame containing 'tokens' column with tokenized text

    Returns:
        DataFrame with additional analysis columns
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # 1. Youth Slang Detection
    youth_slang = set([
        'lit', 'fire', 'dope', 'sick', 'woke', 'slay', 'vibe', 'vibey', 'flex', 'cap', 'no cap',
        'bet', 'based', 'bussin', 'slaps', 'banger', 'goat', 'fam', 'stan', 'chill', 'sus',
        'snatched', 'tea', 'bomb', 'yeet', 'rizz', 'yolo', 'fomo', 'smh', 'legit', 'extra',
        'goals', 'savage', 'hype', 'low-key', 'high-key', 'shook', 'squad', 'crew', 'vibin',
        'mood', 'periodt', 'wig', 'snapped', 'iconic', 'glow-up', 'fit', 'drip', 'on point',
        'hits different', 'simp', 'main character', 'energy', 'meme', 'mood', 'salty', 'basic',
        'ghosted', 'triggered', 'adulting', 'lowkey', 'highkey', 'clout', 'gucci', 'spill', 'vibe check',
        'cancelled', 'bruh', 'bro', 'sis', 'pressed', 'deadass', 'finesse', 'flex', 'gassed',
        'kiki', 'on god', 'sheesh', 'slatt', 'snatched', 'tweakin', 'weak', 'hits', 'send it',
        'straight fire', 'facts', 'cap', 'clapped', 'beast', 'goes hard', 'throw shade',
        'living rent free', 'main character energy', 'rent free', 'catch these hands', 'finna',
        'boujee', 'clown', 'cringe', 'feels', 'shade', 'wrecked', 'yassss', 'sliving', 'skkrt',
        'fyre', 'fleek', 'bae', 'bestie', 'bestfriend', 'bestie', 'besties', 'bestfriends', 'baddie',
        'glizzy', 'clutch', 'rent free', 'cheugy', 'oop', 'period', 'sksksk', 'sliving', 'skkrt',
        'tea', 'thirsty', 'turnt', 'vibe', 'vibin', 'vibing', 'understood the assignment',
        'unalive', 'ate', 'ate that', 'serve', 'served', 'slay', 'slayed', 'snatched', 'ick',
        'mid', 'stan', 'main character', 'mothered', 'zaddy', 'slept on', 'ratioed', 'ratio',
        'gang', 'gang gang', 'demure', 'purr', 'sheddy', 'shady', 'no printer', 'fr', 'frfr',
        'asl', 'gyat', 'lowkey', 'highkey', 'deadass', 'deadazz', 'npc', 'devious lick', 'jit',
        'jawn', 'lewk', 'no cap', 'ong', 'yapping', 'goofy', 'glizzy', 'fit check', 'opp',
        'plugged in', 'plug', 'plug walk', 'pushed p', 'pushing p', 'skibidi', 'sigma', 'alpha',
        'bbg', 'bbygirl', 'caught in 4k', 'finna', 'cop', 'copped', 'skrrt', 'drip', 'drippy',
        'saucy', 'sauce', 'valid', 'ate down', 'mother', 'mothered', 'material gworl',
        'gaslight', 'gatekeep', 'girlboss', 'gamer', 'gc', 'okurr', 'omg', 'sml', 'szn',
        'built different', 'hits different', 'just vibes', 'emo', 'awks', 'main pop girl', 'pop off'
    ])

    df['slang_word_count'] = df['tokens'].apply(lambda tokens: sum(1 for t in tokens if t in youth_slang))
    df['slang_ratio'] = df['slang_word_count'] / df['tokens'].apply(len).replace(0, 1)

    # 2. Flavor Reference Analysis
    flavor_words = set([
        'sweet', 'candy', 'dessert', 'chocolate', 'vanilla', 'strawberry', 'banana', 'cherry',
        'watermelon', 'apple', 'grape', 'peach', 'mango', 'kiwi', 'fruity', 'tropical', 'berry',
        'blueberry', 'raspberry', 'caramel', 'cream', 'creamy', 'custard', 'pie', 'cake', 'tart',
        'cookie', 'cookies', 'donut', 'cereal', 'milkshake', 'smoothie', 'juice', 'bubblegum',
        'cotton', 'taffy', 'lollipop', 'popsicle', 'gummy', 'sour', 'sweet', 'sugary', 'honey',
        'tart', 'tangy', 'citrus', 'orange', 'lemon', 'lime', 'grapefruit', 'pineapple', 'coconut',
        'mint', 'menthol', 'cinnamon', 'spice', 'spicy', 'nutmeg', 'clove', 'coffee', 'espresso',
        'mocha', 'latte', 'cappuccino', 'tea', 'chai', 'cola', 'soda', 'fizz', 'fizzy', 'pop',
        'slushie', 'slurpee', 'ice', 'iced', 'chill', 'chilled', 'crisp', 'refreshing', 'cool',
        'melon', 'honeydew', 'cantaloupe', 'pomegranate', 'plum', 'apricot', 'nectarine', 'peach',
        'fig', 'date', 'raisin', 'cranberry', 'currant', 'passionfruit', 'guava', 'papaya', 'dragonfruit',
        'starfruit', 'jackfruit', 'lychee', 'rambutan', 'durian', 'persimmon', 'quince', 'kumquat',
        'tangerine', 'mandarin', 'clementine', 'yuzu', 'acai', 'goji', 'mulberry', 'elderberry',
        'boysenberry', 'gooseberry', 'blackberry', 'blackcurrant', 'açaí', 'gelato', 'sherbet',
        'sorbet', 'parfait', 'sundae', 'brownie', 'fudge', 'ganache', 'frosting', 'icing', 'glaze',
        'syrup', 'drizzle', 'sauce', 'compote', 'jam', 'jelly', 'preserve', 'curd', 'marmalade',
        'brittle', 'toffee', 'nougat', 'marshmallow', 'licorice', 'anise', 'marzipan', 'butterscotch',
        'maple', 'pancake', 'waffle', 'danish', 'pastry', 'croissant', 'macaron', 'éclair', 'truffle',
        'bonbon', 'praline', 'divinity', 'flan', 'pudding', 'mousse', 'cheesecake', 'tiramisu',
        'crème', 'brûlée', 'panna', 'cotta', 'snickerdoodle', 'biscuit', 'scone', 'muffin', 'cupcake',
        'cobbler', 'crisp', 'crumble', 'strudel', 'danish', 'cinnamon roll', 'froyo', 'yogurt',
        'icebox', 'crepe', 'churro', 'cannoli', 'baklava', 'ambrosia', 'pudding', 'sweet', 'treat',
        'confection', 'indulgence', 'delicacy', 'pleasure', 'guilty pleasure', 'sinful', 'decadent'
    ])

    df['flavor_word_count'] = df['tokens'].apply(lambda tokens: sum(1 for t in tokens if t in flavor_words))
    df['flavor_ratio'] = df['flavor_word_count'] / df['tokens'].apply(len).replace(0, 1)

    # 3. Social Media Reference Analysis
    social_media_terms = set([
        'follow', 'like', 'share', 'post', 'story', 'stories', 'comment', 'hashtag', 'trending',
        'viral', 'challenge', 'tiktok', 'instagram', 'insta', 'snap', 'tweet', 'dm', 'live',
        'stream', 'subscribe', 'notification', 'filter', 'profile', 'tag', 'mention', 'trend',
        'repost', 'retweet', 'selfie', 'influencer', 'collab', 'collaboration', 'community',
        'follower', 'following', 'feed', 'timeline', 'viral', 'reel', 'short', 'playlist',
        'thread', 'bio', 'avatar', 'handle', 'username', 'verified', 'blue check', 'trending',
        'explore', 'discover', 'algorithm', 'content', 'creator', 'discord', 'server', 'channel',
        'youtube', 'twitch', 'stream', 'streamer', 'gaming', 'gamer', 'esports', 'tournament',
        'highlight', 'clip', 'edit', 'montage', 'compilation', 'duet', 'stitch', 'greenscreen',
        'filter', 'ar', 'effect', 'soundbite', 'sound', 'audio', 'remix', 'meme', 'gif', 'emote',
        'emoji', 'react', 'reaction', 'fanbase', 'stan', 'cancel', 'cancellation', 'drama',
        'expose', 'exposed', 'tea', 'receipts', 'callout', 'link in bio', 'swipe up', 'newsletter',
        'subscribe', 'notification', 'bell', 'turned on', 'live notification', 'premiere',
        'facebook', 'twitter', 'snapchat', 'youtube', 'tiktok', 'reddit', 'pinterest', 'linkedin',
        'tumblr', 'whatsapp', 'telegram', 'discord', 'spotify', 'apple music', 'soundcloud',
        'subreddit', 'thread', 'forum', 'board', 'group', 'page', 'event', 'invite', 'request',
        'accept', 'decline', 'block', 'mute', 'ghost', 'ghosting', 'seen', 'read', 'message',
        'inbox', 'chat', 'group chat', 'livestream', 'going live', 'crosspost', 'pinned',
        'bookmark', 'saved', 'collection', 'album', 'gallery', 'story highlight', 'swipe',
        'click', 'tap', 'double tap', 'scroll', 'refresh', 'update', 'engagement', 'impression',
        'analytics', 'reach', 'viral', 'breakout', 'blow up', 'famous', 'influencer', 'content creator',
        'blue tick', 'check mark', 'upload', 'download', 'screengrab', 'screenshot', 'screen recording',
        'pinch', 'zoom', 'filter', 'effect', 'animation', 'sticker', 'caption', 'edit', 'crop',
        'layout', 'grid', 'aesthetic', 'theme', 'feed goals', 'interface', 'screenshot', 'screengrab'
    ])

    df['social_media_count'] = df['tokens'].apply(lambda tokens: sum(1 for t in tokens if t in social_media_terms))
    df['social_media_ratio'] = df['social_media_count'] / df['tokens'].apply(len).replace(0, 1)

    # 4. Urgency/FOMO Language Detection
    urgency_words = set([
        'now', 'hurry', 'limited', 'exclusive', 'only', 'today', 'don\'t miss', 'don\'t wait',
        'quick', 'fast', 'instant', 'immediately', 'asap', 'rush', 'soon', 'before', 'fomo',
        'missing', 'miss out', 'chance', 'opportunity', 'deal', 'special', 'offer', 'discount',
        'urgent', 'emergency', 'critical', 'crucial', 'vital', 'essential', 'important', 'necessary',
        'priority', 'top priority', 'immediate', 'instant', 'rapid', 'swift', 'speedy', 'expedited',
        'accelerated', 'brief', 'fleeting', 'momentary', 'temporary', 'transient', 'ephemeral',
        'short-lived', 'short-term', 'brief', 'passing', 'transitory', 'temporary', 'impermanent',
        'time-limited', 'time-sensitive', 'deadline', 'cutoff', 'countdown', 'ticking clock',
        'running out', 'almost gone', 'selling out', 'going fast', 'hot item', 'in demand',
        'popular', 'trending', 'selling quickly', 'while supplies last', 'while stocks last',
        'until supplies run out', 'once they\'re gone', 'won\'t last', 'won\'t last long',
        'last chance', 'final opportunity', 'final chance', 'never again', 'one time only',
        'once in a lifetime', 'rare opportunity', 'rare chance', 'don\'t delay', 'act now',
        'act fast', 'act quickly', 'act immediately', 'move fast', 'respond now', 'respond quickly',
        'respond immediately', 'get it now', 'buy now', 'shop now', 'order now', 'purchase now',
        'claim now', 'reserve now', 'book now', 'schedule now', 'register now', 'sign up now',
        'join now', 'start now', 'begin now', 'enroll now', 'apply now', 'try now', 'test now',
        'sample now', 'preview now', 'view now', 'watch now', 'listen now', 'download now',
        'access now', 'unlock now', 'get started now', 'begin today', 'start today', 'don\'t wait',
        'can\'t wait', 'shouldn\'t wait', 'mustn\'t wait', 'no waiting', 'no delay', 'instant access',
        'instant delivery', 'instant results', 'instant gratification', 'immediate results',
        'immediate benefits', 'immediate advantages', 'immediate rewards', 'immediate perks',
        'immediate gains', 'immediate improvements', 'immediate upgrades', 'immediate enhancements',
        'right away', 'right now', 'this minute', 'this second', 'this instant', 'as we speak',
        'before it\'s too late', 'before it\'s gone', 'before they\'re all gone', 'before someone else',
        'before your peers', 'before your friends', 'before your colleagues', 'before your competitors',
        'before prices increase', 'before rates go up', 'going, going, gone', 'hot deal', 'hot offer',
        'flash sale', 'lightning deal', 'limited time', 'limited edition', 'limited release',
        'limited availability', 'limited quantity', 'limited stock', 'limited supply', 'limited inventory',
        'limited production', 'limited distribution', 'expiring soon', 'expires soon', 'ending soon'
    ])

    df['urgency_count'] = df['tokens'].apply(lambda tokens: sum(1 for t in tokens if t in urgency_words))
    df['urgency_ratio'] = df['urgency_count'] / df['tokens'].apply(len).replace(0, 1)

    # 5. Identity/Belonging Language
    identity_words = set([
        'generation', 'community', 'culture', 'lifestyle', 'identity', 'rebel', 'revolution',
        'movement', 'join', 'belong', 'member', 'exclusive', 'club', 'squad', 'crew', 'team',
        'family', 'gang', 'tribe', 'circle', 'group', 'represent', 'authentic', 'real', 'true',
        'genuine', 'loyal', 'faithful', 'dedicated', 'committed', 'devoted', 'allegiance',
        'solidarity', 'unity', 'together', 'collective', 'alliance', 'coalition', 'fellowship',
        'brotherhood', 'sisterhood', 'kinship', 'bond', 'connection', 'relationship', 'affiliation',
        'association', 'network', 'society', 'organization', 'clique', 'posse', 'entourage',
        'followers', 'fanbase', 'fans', 'supporters', 'advocates', 'enthusiasts', 'devotees',
        'believers', 'champions', 'ambassadors', 'representatives', 'spokespeople', 'voices',
        'faces', 'icons', 'symbols', 'figureheads', 'leaders', 'pioneers', 'trailblazers',
        'trendsetters', 'influencers', 'tastemakers', 'style-makers', 'role models', 'heroes',
        'idols', 'stars', 'celebrities', 'vips', 'elites', 'insiders', 'inner circle', 'chosen',
        'selected', 'privileged', 'special', 'unique', 'distinct', 'different', 'standout',
        'exceptional', 'extraordinary', 'remarkable', 'notable', 'significant', 'important',
        'valued', 'appreciated', 'recognized', 'acknowledged', 'respected', 'admired', 'esteemed',
        'revered', 'venerated', 'honored', 'celebrated', 'acclaimed', 'lauded', 'praised',
        'commended', 'approved', 'endorsed', 'supported', 'backed', 'upheld', 'championed',
        'defended', 'protected', 'sheltered', 'safeguarded', 'secured', 'ensured', 'guaranteed',
        'assured', 'confident', 'certain', 'sure', 'definite', 'absolute', 'unquestionable',
        'undeniable', 'indisputable', 'irrefutable', 'incontrovertible', 'undoubted', 'validated',
        'verified', 'confirmed', 'corroborated', 'substantiated', 'authenticated', 'legitimized',
        'acceptance', 'acknowledgment', 'recognition', 'validation', 'affirmation', 'confirmation',
        'reassurance', 'approval', 'endorsement', 'support', 'backing', 'advocacy', 'promotion',
        'encouragement', 'motivation', 'inspiration', 'empowerment', 'enablement', 'facilitation',
        'fostering', 'nurturing', 'cultivation', 'development', 'growth', 'progression', 'advancement'
    ])

    df['identity_count'] = df['tokens'].apply(lambda tokens: sum(1 for t in tokens if t in identity_words))
    df['identity_ratio'] = df['identity_count'] / df['tokens'].apply(len).replace(0, 1)

    # 6. Novelty/Excitement Language
    excitement_words = set([
        'new', 'latest', 'innovative', 'revolutionary', 'breakthrough', 'discover', 'unveil',
        'introducing', 'exciting', 'amazing', 'incredible', 'awesome', 'wow', 'mind-blowing',
        'game-changing', 'next-level', 'epic', 'legendary', 'insane', 'crazy', 'wild', 'extreme',
        'intense', 'powerful', 'strong', 'potent', 'effective', 'efficient', 'productive', 'successful',
        'thrilling', 'exhilarating', 'electrifying', 'stimulating', 'invigorating', 'energizing',
        'revitalizing', 'refreshing', 'rejuvenating', 'renewing', 'restoring', 'reviving',
        'revamping', 'reimagining', 'reinventing', 'redefining', 'reshaping', 'transforming',
        'converting', 'altering', 'changing', 'modifying', 'adapting', 'adjusting', 'updating',
        'upgrading', 'enhancing', 'improving', 'augmenting', 'amplifying', 'magnifying',
        'intensifying', 'heightening', 'escalating', 'accelerating', 'expediting', 'quickening',
        'hastening', 'speeding', 'boosting', 'elevating', 'lifting', 'raising', 'increasing',
        'expanding', 'extending', 'enlarging', 'broadening', 'widening', 'deepening', 'strengthening',
        'reinforcing', 'fortifying', 'solidifying', 'stabilizing', 'securing', 'firming',
        'toughening', 'hardening', 'optimizing', 'maximizing', 'perfecting', 'refining',
        'polishing', 'fine-tuning', 'customizing', 'tailoring', 'personalizing', 'adjusting',
        'vibrant', 'vivid', 'brilliant', 'radiant', 'luminous', 'gleaming', 'glowing', 'blazing',
        'flaming', 'burning', 'scorching', 'sizzling', 'red-hot', 'fiery', 'passionate', 'ardent',
        'fervent', 'zealous', 'enthusiastic', 'eager', 'keen', 'avid', 'excited', 'thrilled',
        'elated', 'ecstatic', 'euphoric', 'overjoyed', 'jubilant', 'exultant', 'triumphant',
        'proud', 'pleased', 'satisfied', 'gratified', 'fulfilled', 'content', 'happy', 'delighted',
        'joyful', 'cheerful', 'gleeful', 'merry', 'blissful', 'rapturous', 'enraptured',
        'entranced', 'captivated', 'fascinated', 'enthralled', 'spellbound', 'mesmerized',
        'hypnotized', 'transfixed', 'absorbed', 'engrossed', 'immersed', 'engaged', 'involved',
        'committed', 'dedicated', 'devoted', 'passionate', 'enthusiastic', 'zealous', 'fervent',
        'ardent', 'intense', 'profound', 'deep', 'heartfelt', 'sincere', 'genuine', 'authentic',
        'real', 'true', 'bona fide', 'legitimate', 'valid', 'credible', 'believable', 'plausible',
        'sensational', 'phenomenal', 'fantastic', 'fabulous', 'marvelous', 'wonderful', 'splendid',
        'magnificent', 'superb', 'excellent', 'exceptional', 'outstanding', 'remarkable', 'notable'
    ])

    df['excitement_count'] = df['tokens'].apply(lambda tokens: sum(1 for t in tokens if t in excitement_words))
    df['excitement_ratio'] = df['excitement_count'] / df['tokens'].apply(len).replace(0, 1)

    # 7. Simplicity Analysis - measuring language complexity
    df['avg_word_length'] = df['tokens'].apply(lambda tokens: np.mean([len(t) for t in tokens]) if tokens else 0)

    # Count number of "long words" (3+ syllables) as a proxy for complexity
    def count_syllables(word):
        word = word.lower()
        if len(word) <= 3:
            return 1
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    df['complex_word_ratio'] = df['tokens'].apply(
        lambda tokens: sum(1 for t in tokens if count_syllables(t) >= 3) / len(tokens) if len(tokens) > 0 else 0
    )

    # 8. "Cool Factor" Language
    cool_factor_words = set([
        'cool', 'awesome', 'rad', 'fresh', 'hot', 'sleek', 'stylish', 'trendy', 'hip',
        'premium', 'elite', 'pro', 'ultimate', 'perfect', 'ideal', 'essential', 'must-have',
        'popular', 'fashionable', 'sophisticated', 'smooth', 'slick', 'clean', 'dope', 'sick',
        'killer', 'wicked', 'epic', 'legendary', 'iconic', 'classic', 'retro', 'vintage', 'old-school',
        'next-gen', 'cutting-edge', 'revolutionary', 'groundbreaking', 'innovative', 'disruptive',
        'game-changing', 'next-level', 'top-tier', 'quality', 'superior', 'exclusive', 'limited',
        'rare', 'unique', 'custom', 'personalized', 'signature', 'special', 'choice', 'select',
        'curated', 'crafted', 'designed', 'engineered', 'built', 'made', 'created', 'developed',
        'authentic', 'genuine', 'real', 'original', 'true', 'legit', 'official', 'certified',
        'validated', 'approved', 'endorsed', 'recommended', 'rated', 'reviewed', 'acclaimed',
        'awarded', 'winning', 'champion', 'master', 'expert', 'specialist', 'professional',
        'unmatched', 'unparalleled', 'incomparable', 'unrivaled', 'peerless', 'supreme',
        'luxurious', 'deluxe', 'high-end', 'upscale', 'fancy', 'elegant', 'chic', 'classy',
        'sophisticated', 'refined', 'polished', 'sleek', 'svelte', 'streamlined', 'aerodynamic',
        'futuristic', 'modern', 'contemporary', 'minimalist', 'maximalist', 'bold', 'distinctive',
        'striking', 'eye-catching', 'head-turning', 'attention-grabbing', 'stunning', 'gorgeous',
        'beautiful', 'attractive', 'appealing', 'alluring', 'captivating', 'mesmerizing',
        'hypnotic', 'magnetic', 'irresistible', 'compelling', 'impressive', 'incredible',
        'amazing', 'astonishing', 'astounding', 'remarkable', 'extraordinary', 'exceptional',
        'outstanding', 'excellent', 'superb', 'magnificent', 'marvelous', 'wonderful', 'fantastic',
        'fabulous', 'terrific', 'tremendous', 'phenomenal', 'mind-blowing', 'mind-bending',
        'breathtaking', 'jaw-dropping', 'awe-inspiring', 'unbelievable', 'inconceivable',
        'unimaginable', 'beyond', 'limitless', 'boundless', 'endless', 'infinite', 'ultimate',
        'paramount', 'foremost', 'preeminent', 'prominent', 'leading', 'dominant', 'pioneering',
        'trailblazing', 'pathfinding', 'groundbreaking', 'revolutionary', 'radical', 'avant-garde',
        'edgy', 'cutting-edge', 'bleeding-edge', 'state-of-the-art', 'next-generation', 'advanced',
        'progressive', 'forward-thinking', 'visionary', 'imaginative', 'creative', 'inventive',
        'ingenious', 'clever', 'brilliant', 'smart', 'intelligent', 'sophisticated', 'complex',
        'intricate', 'detailed', 'precise', 'accurate', 'exact', 'flawless', 'impeccable', 'pristine'
    ])

    df['cool_factor_count'] = df['tokens'].apply(lambda tokens: sum(1 for t in tokens if t in cool_factor_words))
    df['cool_factor_ratio'] = df['cool_factor_count'] / df['tokens'].apply(len).replace(0, 1)

    # 9. Calculate TF-IDF to find distinctive terms by brand
    # This helps identify brand-specific terminology that might be targeting youth
    def get_token_string(tokens):
        return ' '.join(tokens) if tokens else ''

    df['token_string'] = df['tokens'].apply(get_token_string)

    # Group by brand to find distinctive terms
    distinctive_term_score = np.zeros(len(df))

    try:
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['token_string'])
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Get distinctive terms for each brand
        brand_distinctive_terms = {}
        for brand in df['brand'].unique():
            brand_mask = df['brand'] == brand
            if sum(brand_mask) > 0:  # Make sure brand has at least one sample
                brand_indices = np.where(brand_mask)[0]
                brand_tfidf = tfidf_matrix[brand_indices].toarray().mean(axis=0)
                sorted_indices = np.argsort(brand_tfidf)[::-1]
                top_terms = [feature_names[i] for i in sorted_indices[:20]]  # Get top 20 terms
                brand_distinctive_terms[brand] = top_terms

        # Add brand-specific distinctive term score
        def distinctive_term_score(tokens, brand):
            if brand not in brand_distinctive_terms or not tokens:
                return 0
            distinctive_terms = set(brand_distinctive_terms[brand])
            return sum(1 for t in tokens if t in distinctive_terms) / len(tokens)

        df['distinctive_term_score'] = df.apply(
            lambda row: distinctive_term_score(row['tokens'], row['brand']), axis=1
        )
    except Exception as e:
        print(f"Error in TF-IDF analysis: {e}")
        # Fallback if TF-IDF analysis fails (e.g., due to empty strings)
        df['distinctive_term_score'] = 0

    # 10. Create Combined Youth Appeal Score with all 10 factors
    # Weights can be adjusted based on what factors you find most predictive
    weights = {
        'slang_ratio': 0.15,
        'flavor_ratio': 0.15,
        'social_media_ratio': 0.1,
        'urgency_ratio': 0.1,
        'identity_ratio': 0.1,
        'excitement_ratio': 0.1,
        'complex_word_ratio': -0.1,  # Negative weight because complex language is less youth-oriented
        'cool_factor_ratio': 0.1,
        'distinctive_term_score': 0.1
    }

    # Initialize youth appeal score
    df['youth_appeal_score'] = 0

    # Apply weights to each factor
    for factor, weight in weights.items():
        if factor in df.columns:
            # Normalize factor values to 0-1 range if needed
            max_val = df[factor].max()
            if max_val > 0:  # Avoid division by zero
                normalized_factor = df[factor] / max_val
                df['youth_appeal_score'] += normalized_factor * weight
            else:
                df['youth_appeal_score'] += df[factor] * weight

    # Flag images based on threshold
    df['text_targets_youth'] = df['youth_appeal_score'] > 0.3  # Adjust threshold as needed

    return df


def analyze_text_complexity(df):
    """
    Analyzes the readability and complexity of text content to identify content
    likely targeting younger audiences (simpler, easier to read text).

    Args:
        df: DataFrame containing 'extracted_text' column

    Returns:
        DataFrame with additional readability metrics
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Define a function to safely calculate reading metrics
    def get_reading_metrics(text):
        if not isinstance(text, str) or not text.strip():
            return {
                'reading_grade': np.nan,
                'reading_age': np.nan,
                'flesch_score': np.nan,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'avg_syllables_per_word': 0,
                'complex_word_percentage': 0
            }

        try:
            # Calculate various readability metrics
            flesch_score = textstat.flesch_reading_ease(text)
            fk_grade = textstat.flesch_kincaid_grade(text)
            reading_age = fk_grade + 5  # Approximate reading age from grade level

            # Get sentence and word statistics
            sentence_count = textstat.sentence_count(text)
            total_words = textstat.lexicon_count(text, removepunct=True)
            syllable_count = textstat.syllable_count(text)
            complex_word_count = textstat.difficult_words(text)

            # Calculate average sentence length and syllables per word
            avg_sentence_length = total_words / max(sentence_count, 1)
            avg_syllables_per_word = syllable_count / max(total_words, 1)
            complex_word_percentage = complex_word_count / max(total_words, 1) * 100

            return {
                'reading_grade': fk_grade,
                'reading_age': reading_age,
                'flesch_score': flesch_score,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'avg_syllables_per_word': avg_syllables_per_word,
                'complex_word_percentage': complex_word_percentage
            }
        except Exception as e:
            print(f"Error analyzing text complexity: {e}")
            return {
                'reading_grade': np.nan,
                'reading_age': np.nan,
                'flesch_score': np.nan,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'avg_syllables_per_word': 0,
                'complex_word_percentage': 0
            }

    # Apply the function to each row
    metrics = df['extracted_text'].apply(get_reading_metrics)

    # Extract metrics into separate columns
    df['reading_grade'] = metrics.apply(lambda x: x['reading_grade'])
    df['reading_age'] = metrics.apply(lambda x: x['reading_age'])
    df['flesch_score'] = metrics.apply(lambda x: x['flesch_score'])
    df['sentence_count'] = metrics.apply(lambda x: x['sentence_count'])
    df['avg_sentence_length'] = metrics.apply(lambda x: x['avg_sentence_length'])
    df['avg_syllables_per_word'] = metrics.apply(lambda x: x['avg_syllables_per_word'])
    df['complex_word_percentage'] = metrics.apply(lambda x: x['complex_word_percentage'])

    # Flag text written at a younger reading level (adjust threshold as needed)
    # Higher Flesch scores indicate easier readability
    df['simple_language_flag'] = df['flesch_score'] > 80  # Very easy to read
    df['youth_reading_level'] = df['reading_grade'] < 8    # Below 8th grade level

    # Score from 0-1 based on how likely the text targets youth based on simplicity
    # Higher scores indicate more youth-friendly text
    df['readability_youth_score'] = (
        ((100 - df['flesch_score'].clip(0, 100)) / -100) +  # Invert so higher = more youth-friendly
        ((15 - df['reading_grade'].clip(0, 15)) / 15) +     # Lower grade level = higher score
        ((10 - df['avg_sentence_length'].clip(0, 10)) / 10) # Shorter sentences = higher score
    ) / 3  # Average the three components

    return df

def detect_emojis_and_specials(df):
    """
    Detects emojis, special characters, and formatting patterns that may
    indicate youth-targeted content.

    Args:
        df: DataFrame containing 'extracted_text' column

    Returns:
        DataFrame with emoji and special character analysis
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    def analyze_special_chars(text):
        if not isinstance(text, str) or not text.strip():
            return {
                'emoji_count': 0,
                'contains_emojis': False,
                'emoji_list': [],
                'excessive_punctuation': False,
                'exclamation_count': 0,
                'question_count': 0,
                'special_char_count': 0,
                'all_caps_words': 0,
                'hashtag_count': 0,
                'at_mention_count': 0,
                'special_formatting': False
            }

        # Count emojis using the emoji module
        emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
        emoji_count = len(emoji_list)

        # Count exclamation and question marks
        exclamation_count = text.count('!')
        question_count = text.count('?')

        # Check for excessive punctuation (repeating ! or ? marks)
        excessive_punctuation = bool(re.search(r'[!?]{2,}', text))

        # Count special characters (excluding normal punctuation)
        special_chars = re.findall(r'[^\w\s.,;:"\'?\-!()]', text)
        special_char_count = len(special_chars)

        # Count ALL CAPS words (often used for emphasis in youth marketing)
        all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))

        # Count hashtags and @mentions (social media style)
        hashtag_count = len(re.findall(r'#\w+', text))
        at_mention_count = len(re.findall(r'@\w+', text))

        # Check for "special" formatting often used in youth marketing
        special_formatting_patterns = [
            r'\*\*.*?\*\*',           # **bold** text
            r'~.*?~',                 # ~strikethrough~ text
            r'_.*?_',                 # _italics_ text
            r'\>\>\>.*?\<\<\<',       # >>>attention<<< markers
            r'<3',                    # <3 hearts
            r'\([0-9]+\)',            # (100) percentage style
            r'\b[A-Z][a-z]*[A-Z]',    # CamelCase or mIxEdCaSe
            r'[a-zA-Z]\.{3}',         # trailing... for suspense
            r'x+o+x+',                # xoxo hugs and kisses
            r'([hH][aA]){2,}'         # hahaha, HAHAHA laughter
        ]

        special_formatting = any(re.search(pattern, text) for pattern in special_formatting_patterns)

        return {
            'emoji_count': emoji_count,
            'contains_emojis': emoji_count > 0,
            'emoji_list': emoji_list,
            'excessive_punctuation': excessive_punctuation,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'special_char_count': special_char_count,
            'all_caps_words': all_caps_words,
            'hashtag_count': hashtag_count,
            'at_mention_count': at_mention_count,
            'special_formatting': special_formatting
        }

    # Apply the analysis function
    results = df['extracted_text'].apply(analyze_special_chars)

    # Extract results into individual columns
    df['emoji_count'] = results.apply(lambda x: x['emoji_count'])
    df['contains_emojis'] = results.apply(lambda x: x['contains_emojis'])
    df['excessive_punctuation'] = results.apply(lambda x: x['excessive_punctuation'])
    df['exclamation_count'] = results.apply(lambda x: x['exclamation_count'])
    df['all_caps_words'] = results.apply(lambda x: x['all_caps_words'])
    df['hashtag_count'] = results.apply(lambda x: x['hashtag_count'])
    df['at_mention_count'] = results.apply(lambda x: x['at_mention_count'])
    df['special_formatting'] = results.apply(lambda x: x['special_formatting'])

    # Calculate a special characters score (0-1) indicating youth-targeting
    max_exclamation = 5  # Cap for normalization
    max_emoji = 5        # Cap for normalization

    df['special_chars_youth_score'] = (
        (df['contains_emojis'].astype(int) * 0.3) +
        (df['emoji_count'].clip(0, max_emoji) / max_emoji * 0.2) +
        (df['excessive_punctuation'].astype(int) * 0.1) +
        (df['exclamation_count'].clip(0, max_exclamation) / max_exclamation * 0.15) +
        (df['hashtag_count'] > 0).astype(int) * 0.1 +
        (df['at_mention_count'] > 0).astype(int) * 0.05 +
        (df['all_caps_words'] > 0).astype(int) * 0.05 +
        (df['special_formatting'].astype(int) * 0.05)
    )

    # Flag content that has a high special characters youth score
    df['special_chars_youth_flag'] = df['special_chars_youth_score'] > 0.4  # Adjust threshold as needed

    return df



def check_warning_presence(df):
    """
    Adds a column indicating whether the token 'warning' is present in each row's tokens.

    Args:
        df: DataFrame containing 'tokens' column with tokenized text

    Returns:
        DataFrame with additional 'contains_warning' column
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Check if 'warning' is in the tokens list
    df['contains_warning'] = df['tokens'].apply(
        lambda tokens: 'Yes' if 'warning' in [t.lower() for t in tokens] else 'No'
    )

    return df


def main(s3_client):
    bucket_name = 'vapewatchers-2025'
    key = 'vape_ocr_results.csv'
    response_vape_ocr_results = s3_client.get_object(Bucket=bucket_name, Key=key)
    df = pd.read_csv(io.BytesIO(response_vape_ocr_results['Body'].read()))
    
    df = tokenize_and_preprocess(df)
    
    # Now analyze the tokens for youth-targeting language
    analyzed_df = comprehensive_token_analysis(df)
    
    df1 = analyze_text_complexity(analyzed_df)
    
    df2 = detect_emojis_and_specials(df1)
    
    ocr_df = check_warning_presence(df2)
    
    print(ocr_df.head())
    
    csv_buffer = io.StringIO()
    ocr_df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key='vape_ocr_text_analysis.csv', Body=csv_buffer.getvalue(), ContentType="text/csv")