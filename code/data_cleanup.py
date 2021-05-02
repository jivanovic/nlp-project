import re as regex


class CleanupData:
    def iterate(self):
        for cleanup_method in [
                               self.remove_urls,
                               self.remove_usernames,
                               self.remove_na,
                               self.remove_special_chars,
                               self.remove_numbers,
                               self.replace_empty]:
            yield cleanup_method

    @staticmethod
    def remove_by_regex(content, regexp):
        content.loc[:, "text"].replace(regexp, "", inplace=True)
        return content

    def replace_by_regex(content, regexp, replacement_string):
        content.loc[:, "text"].replace(regexp, replacement_string, inplace=True)
        return content

    def remove_urls(self, content):
        return CleanupData.replace_by_regex(content, regex.compile(r"http.?://[^\s]+[\s]?"), 'url')

    def remove_na(self, content):
        return content[content["text"] != "Not Available"]

    def replace_emojis(self, content):  # it unrolls the hashtags to normal words
        content.loc[:, "text"].replace(':)', 'emoji', inplace=True)
        content.loc[:, "text"].replace(':D', 'emoji', inplace=True)
        content.loc[:, "text"].replace(':(', 'emoji', inplace=True)
        return content

    def replace_empty(self, content):  # it unrolls the hashtags to normal words
        content.loc[:, "text"].replace('', 'empty string', inplace=True)
        return content

    def remove_special_chars(self, content):  # it unrolls the hashtags to normal words
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!", "?", ".", "'",
                                                                     "--", "---", "#"]):
            content.loc[:, "text"].replace(remove, "", inplace=True)
        return content

    def remove_usernames(self, content):
        return CleanupData.remove_by_regex(content, regex.compile(r"@[^\s]+[\s]?"))

    def remove_numbers(self, content):
        return CleanupData.remove_by_regex(content, regex.compile(r"\s?[0-9]+\.?[0-9]*"))

'''
import re as regex


class CleanupTwitterData:
    def iterate(self):
        for cleanup_method in [self.remove_urls,
                               self.replace_emojis,
                               self.remove_usernames,
                               self.remove_numbers]:
            yield cleanup_method
    # self.remove_special_chars,
    @staticmethod
    def remove_by_regex(content, regexp):
        content.loc[:, "text"].replace(regexp, "", inplace=True)
        return content

    def replace_by_regex(content, regexp, replacement_string):
        content.loc[:, "text"].replace(regexp, replacement_string, inplace=True)
        return content

    def remove_urls(self, content):
        return CleanupTwitterData.replace_by_regex(content, regex.compile(r"http.?://[^\s]+[\s]?"), 'url')

    def replace_emojis(self, content):  # it unrolls the hashtags to normal words
        content.loc[:, "text"].replace(':)', 'emoji', inplace=True)
        content.loc[:, "text"].replace(':D', 'emoji', inplace=True)
        content.loc[:, "text"].replace(':(', 'emoji', inplace=True)
        return content

    def remove_special_chars(self, content):  # it unrolls the hashtags to normal words
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!", "?", ".", "'",
                                                                     "--", "---", "#"]):
            content.loc[:, "text"].replace(remove, "", inplace=True)
        return content

    def remove_usernames(self, content):
        return CleanupTwitterData.remove_by_regex(content, regex.compile(r"@[^\s]+[\s]?"))

    def remove_numbers(self, content):
        return CleanupTwitterData.remove_by_regex(content, regex.compile(r"\s?[0-9]+\.?[0-9]*"))

'''