import sys

def viterbi_lgt(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    if len1 == 0 or len2 == 0:
        return {}
    # initialize
    viterbi_table = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i-1] == str2[j-1]:
                viterbi_table[i][j] = viterbi_table[i-1][j-1] + 1
            else:
                viterbi_table[i][j] = max(viterbi_table[i-1][j], viterbi_table[i][j-1])

    max_length = viterbi_table[len1][len2]

    # use back track and recursion to get all LGT
    def traceback(i, j):
        if i == 0 or j == 0:
            return {""}

        # if the current token is the same
        if str1[i-1] == str2[j-1]:
            lcs_set = traceback(i - 1, j - 1)
            return {lgt + str1[i - 1] for lgt in lcs_set}

        # if the current token is not the same
        lgt_set = set()
        if viterbi_table[i][j] == viterbi_table[i-1][j]:
            lgt_set.update(traceback(i-1, j))
        if viterbi_table[i][j] == viterbi_table[i][j-1]:
            lgt_set.update(traceback(i, j-1))
        return lgt_set

    return traceback(len1, len2)

    # i, j = len1, len2
    # lgt_set = set()
    # LGT = ""
    # while i > 0 and j > 0:
    #     if str1[i-1] == str2[j-1] and viterbi_table[i][j] == viterbi_table[i-1][j-1] + 1:
    #         LGT = str1[i-1] + LGT
    #         i = i-1
    #         j = j-1
    #         continue
    #     if viterbi_table[i][j] == viterbi_table[i-1][j]:
    #         i = i - 1
    #         continue
    #     if viterbi_table[i][j] == viterbi_table[i][j-1]:
    #         j = j - 1
    #         continue
    # print(LGT)
    # return(LGT)

if __name__ == "__main__":
    str1 = sys.argv[1]
    str2 = sys.argv[2]
    results = viterbi_lgt(str1, str2)
    # print(results)
    for res in results:
        print(res)

