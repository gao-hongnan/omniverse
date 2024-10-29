class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_hashmap = self.get_string_hashmap(string=p)

        output: List[int] = []

        window_size: int = len(p)

        len_s = len(s)
        for index in range(len_s):
            if index == len_s:
                break

            substring = s[index:window_size]
            window_size += 1

            substring_hashmap = self.get_string_hashmap(string=substring)

            if substring_hashmap == p_hashmap:
                # a match
                output.append(index)
        return output


    def get_string_hashmap(self, string: str) -> Dict[str, int]:
        counter = {}

        for char in string:
            if char not in counter:
                counter[char] = 1
            else:
                counter[char] += 1
        return counter