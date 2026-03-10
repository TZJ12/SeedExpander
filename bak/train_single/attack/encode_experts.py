# ciphers we used
import base64

self_chinese_alphabet = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸","子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥","天","地","人","黄"]
english_alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
chinese_alphabet = ["e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z", "a","b","c","d"]

# for i in range(26):
#     print(english_alphabet[i], "->", chinese_alphabet[i])

class Base64Expert():
    def encode(self, s):
        # 对输入字符串进行 Base64 编码
        return base64.b64encode(s.encode('utf-8')).decode('ascii')

    def decode(self, s):
        try:
            # 对 Base64 字符串进行解码
            return base64.b64decode(s).decode('utf-8')
        except Exception as e:
            print(f"Base64 解码失败: {e}")
            return s  # 返回原始字符串以防出错


class SelfDefineCipher():

    def encode(self, s):
        s = s.lower()

        ans = ""
        for letter in s:
            try:
                ans += self_chinese_alphabet[ord(letter.lower()) - 96-1]
            except:
                ans += letter
        return ans

    def decode(self, s):
        ans = ""
        for letter in s:
            try:
                position = self_chinese_alphabet.index(letter)
                ans += english_alphabet[position]
            except:
                ans += letter
        return ans

shift = 3
class CaesarExpert():

    def encode(self, s):
        ans = ''
        for p in s:
            if 'a' <= p <= 'z':
                ans += chr(ord('a') + (ord(p) - ord('a') + shift) % 26)
            elif 'A' <= p <= 'Z':
                ans += chr(ord('A') + (ord(p) - ord('A') + shift) % 26)
            else:
                ans += p

        return ans

    def decode(self, s):
        ans = ''
        for p in s:
            if 'a' <= p <= 'z':
                ans += chr(ord('a') + (ord(p) - ord('a') - shift) % 26)
            elif 'A' <= p <= 'Z':
                ans += chr(ord('A') + (ord(p) - ord('A') - shift) % 26)
            else:
                ans += p
        return ans


class UnicodeExpert():

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("unicode_escape"))
                if len(byte_s) > 8:
                    ans += byte_s[3:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    def decode(self, s):
        ans = bytes(s, encoding="utf8").decode("unicode_escape")
        return ans


class BaseExpert():

    def encode(self, s):
        return s

    def decode(self, s):
        return s


class UTF8Expert():

    def encode(self, s):
        if not s:  # 处理空字符串情况
            return ""
            
        ans = ''
        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("utf8"))
                if len(byte_s) > 8:
                    ans += byte_s[2:-1]
                elif len(byte_s) >= 3:  # 确保字符串长度足够
                    ans += byte_s[-2]
                else:
                    ans += c  # 如果编码异常，保留原字符
            ans += "\n"
        return ans.strip()  # 移除末尾多余的换行符

    def decode(self, s):
        if not s:  # 处理空字符串情况
            return ""
            
        ans = b''
        try:
            while len(s):
                if s.startswith("\\x"):
                    try:
                        ans += bytes.fromhex(s[2:4])
                        s = s[4:]
                    except ValueError:
                        # 如果十六进制转换失败，跳过这个序列
                        ans += bytes(s[0], encoding="utf8")
                        s = s[1:]
                else:
                    ans += bytes(s[0], encoding="utf8")
                    s = s[1:]

            try:
                ans = ans.decode("utf8")
            except UnicodeDecodeError:
                # 如果UTF-8解码失败，尝试使用其他编码或返回原始字节的字符串表示
                ans = str(ans)
            return ans
        except Exception as e:
            print(f"UTF8 解码错误: {e}")
            return s  # 出错时返回原始字符串


class AsciiExpert():

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                try:
                    ans += str(ord(c)) + " "
                except:
                    ans += c
            ans += "\n"
        return ans

    def decode(self, s):
        ans = ""
        lines = s.split("\n")
        for line in lines:
            cs = line.split()
            for c in cs:
                try:
                    ans += chr(int(c))
                except:
                    ans += c
        return ans

class GBKExpert():

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("GBK"))
                if len(byte_s) > 8:
                    ans += byte_s[2:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    def decode(self, s):
        ans = b''
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="GBK")
                s = s[1:]

        ans = ans.decode("GBK")
        return ans


class MorseExpert():

    def encode(self, s):
        s = s.upper()
        MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                           'C': '-.-.', 'D': '-..', 'E': '.',
                           'F': '..-.', 'G': '--.', 'H': '....',
                           'I': '..', 'J': '.---', 'K': '-.-',
                           'L': '.-..', 'M': '--', 'N': '-.',
                           'O': '---', 'P': '.--.', 'Q': '--.-',
                           'R': '.-.', 'S': '...', 'T': '-',
                           'U': '..-', 'V': '...-', 'W': '.--',
                           'X': '-..-', 'Y': '-.--', 'Z': '--..',
                           '1': '.----', '2': '..---', '3': '...--',
                           '4': '....-', '5': '.....', '6': '-....',
                           '7': '--...', '8': '---..', '9': '----.',
                           '0': '-----', ', ': '--..--', '.': '.-.-.-',
                           '?': '..--..', '/': '-..-.', '-': '-....-',
                           '(': '-.--.', ')': '-.--.-'}
        cipher = ''
        lines = s.split("\n")
        for line in lines:
            for letter in line:
                try:
                    if letter != ' ':
                        cipher += MORSE_CODE_DICT[letter] + ' '
                    else:
                        cipher += ' '
                except:
                    cipher += letter + ' '
            cipher += "\n"
        return cipher

    def decode(self, s):
        MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                           'C': '-.-.', 'D': '-..', 'E': '.',
                           'F': '..-.', 'G': '--.', 'H': '....',
                           'I': '..', 'J': '.---', 'K': '-.-',
                           'L': '.-..', 'M': '--', 'N': '-.',
                           'O': '---', 'P': '.--.', 'Q': '--.-',
                           'R': '.-.', 'S': '...', 'T': '-',
                           'U': '..-', 'V': '...-', 'W': '.--',
                           'X': '-..-', 'Y': '-.--', 'Z': '--..',
                           '1': '.----', '2': '..---', '3': '...--',
                           '4': '....-', '5': '.....', '6': '-....',
                           '7': '--...', '8': '---..', '9': '----.',
                           '0': '-----', ', ': '--..--', '.': '.-.-.-',
                           '?': '..--..', '/': '-..-.', '-': '-....-',
                           '(': '-.--.', ')': '-.--.-'}
        decipher = ''
        citext = ''
        lines = s.split("\n")
        for line in lines:
            for letter in line:
                while True and len(letter):
                    if letter[0] not in ['-', '.', ' ']:
                        decipher += letter[0]
                        letter = letter[1:]
                    else:
                        break
                try:
                    if (letter != ' '):
                        i = 0
                        citext += letter
                    else:
                        i += 1
                        if i == 2:
                            decipher += ' '
                        else:
                            decipher += list(MORSE_CODE_DICT.keys())[list(MORSE_CODE_DICT
                                                                          .values()).index(citext)]
                            citext = ''
                except:
                    decipher += letter
            decipher += '\n'
        return decipher



class AtbashExpert():

    def encode(self, text):
        ans = ''
        for s in text:
            try:
                if 'a' <= s <= 'z':
                    # 小写字母: a->z, b->y, ...
                    ans += chr(ord('z') - (ord(s) - ord('a')))
                elif 'A' <= s <= 'Z':
                    # 大写字母: A->Z, B->Y, ...
                    ans += chr(ord('Z') - (ord(s) - ord('A')))
                else:
                    ans += s
            except:
                ans += s
        return ans

    def decode(self, text):
        # Atbash是对称的，解密和加密使用相同的算法
        return self.encode(text)


encode_expert_dict = {
    "unchange": BaseExpert(),
    "baseline": BaseExpert(),
    "caesar": CaesarExpert(),
    "unicode": UnicodeExpert(),
    "morse": MorseExpert(),
    "atbash": AtbashExpert(),
    "utf": UTF8Expert(),
    "ascii": AsciiExpert(),
    "gbk": GBKExpert(),
    "selfdefine": SelfDefineCipher(),
    "base64": Base64Expert(),
}
