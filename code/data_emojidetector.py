class EmojiDetector:
    emojis = {}

    def __init__(self, emoticon_file="data/emojis.txt"):
        from pathlib import Path
        content = Path(emoticon_file).read_text()
        positive = True
        for line in content.split("\n"):
            if "positive" in line.lower():
                positive = True
                continue
            elif "negative" in line.lower():
                positive = False
                continue

            self.emojis[line] = positive

    def is_positive(self, emoticon):
        if emoticon in self.emojis:
            return self.emojis[emoticon]
        return False

    def is_emoticon(self, to_check):
        return to_check in self.emojis
