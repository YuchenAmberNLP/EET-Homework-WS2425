import sys

def longest_common_subsequence(s1, s2):
    l1 = len(s1)
    l2 = len(s2)

    store_string_matrix = [[0 for i in range(l2+1)] for j in range(l1+1)] # (l1 + 1) x (l2 + 1) matrix
                                                                        # row = l1, column = l2
    # print(store_string_matrix)

    # fill 2 strings in the matrix
    for i in range(1, l1+1): # start from row 1, column 1
        for j in range(1, l2+1):
            if s1[i-1] == s2[j-1]: # if the characters are the same
                store_string_matrix[i][j] = store_string_matrix[i-1][j-1] + 1 # add 1 to the diagonal value
            else:
                store_string_matrix[i][j] = max(store_string_matrix[i-1][j], store_string_matrix[i][j-1]) # take the maximum value from the top or left

    # print(store_string_matrix)
    return store_string_matrix

# backtracking
def get_lcs(s1, s2, store_string_matrix, i, j):
    # l1 = len(s1)
    # l2 = len(s2)
    # result = []
    # i = l1 # start from the last row
    # j = l2 # start from the last column
    if i == 0 or j == 0: # one of the strings has been completely processed 
        return {""} # a set containing an empty string
    
    if s1[i-1] == s2[j-1]:
        result = get_lcs(s1, s2, store_string_matrix, i-1, j-1)# recursive call from the diagonal value
        return {subset + s1[i-1] for subset in result} 

    else:
        result = set() # create a set
        if store_string_matrix[i-1][j] >= store_string_matrix[i][j-1]:
            # Move up
            result.update(get_lcs(s1, s2, store_string_matrix, i-1, j)) # recurive call from the up value & add the character from up to the result
        if store_string_matrix[i][j-1] >= store_string_matrix[i-1][j]:
            # Move left
            result.update(get_lcs(s1, s2, store_string_matrix, i, j-1)) # recurive call from the left value & add the character from left to the result
        return result



# l1 = 'AGGTAB' # 6
# l2 = 'GXTXAYB' # 7
# store_string_matrix = longest_common_subsequence(l1, l2)
# print(get_lcs(l1, l2, store_string_matrix)) # GTAB

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Invalid number of arguments")
        sys.exit(1)
    
    l1 = sys.argv[1]
    l2 = sys.argv[2]

    store_string_matrix = longest_common_subsequence(l1, l2)
    result = get_lcs(l1, l2, store_string_matrix, len(l1), len(l2))
    print(f"Longest Common Subsequences between {l1} and {l2}: ")
    for subset in result:
        print(subset)
