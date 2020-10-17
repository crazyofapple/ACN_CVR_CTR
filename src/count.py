'''
cephfs conut clicked ads length and converted ads length
'''
# path = "./pdata_20190506_all.csv"

path = '../data/cvr_data/small_tiny_v2.txt'
HEADER = ["C39", "C38", "C35", "C34", "C37", "C36", "C31", "C30", "C33", "C32", "C153", "C152", "C151", "C150", "C157",
          "C156", "C155", "C154", "C159", "C158", "C22", "C23", "C20", "C21", "C26", "C27", "C24", "C25", "C28", "C168",
          "C169", "C166", "C167", "C164", "C165", "C162", "C163", "C160", "C161", "C184", "C57", "C56", "C55", "C54",
          "C53", "C52", "C51", "C50", "C188", "C189", "C59", "C58", "C179", "C178", "C141", "C133", "C186", "C171",
          "C170", "C173", "C172", "C175", "C174", "C177", "C176", "C191", "C44", "C45", "C46", "C47", "C40", "C41",
          "C42", "C43", "C48", "C49", "C190", "C121", "C108", "C109", "C104", "C105", "C106", "C100", "C101", "C102",
          "C103", "C79", "C78", "C185", "C71", "C70", "C73", "C72", "C75", "C74", "C77", "C187", "C180", "C181", "C182",
          "C183", "C9", "C8", "C119", "C118", "C116", "C1", "C114", "C7", "C6", "C5", "C4", "C140", "C68", "C69", "C66",
          "C67", "C64", "C65", "C62", "C63", "C60", "C61", "C143", "C144", "C115", "C145", "C113", "C147", "C112",
          "C148", "C128", "C129", "C111", "C149", "C122", "C123", "C120", "C110", "C126", "C127", "C124", "C125", "C19",
          "C18", "C13", "C12", "C11", "C10", "C17", "C16", "C15", "C14", "C93", "C92", "C91", "C90", "C97", "C96",
          "C95", "C94", "C99", "C98", "C139", "C138", "C135", "C134", "C137", "C136", "C131", "C130", "C80", "C81",
          "C82", "C83", "C84", "C85", "C146", "C87", "C88", "C89", "C132", "C142", "target_cvr", "target_ctr", "id",
          "type_label"]
sparse_features = ['C39', 'C7', 'C28', 'C5', 'C30', 'C14', 'C32', 'C18', 'C27', 'C24', 'C13', 'C9', 'C36', 'C48',
                   'C12', 'C49', 'C33', 'C8', 'C22', 'C4', 'C23', 'C16', 'C35', 'C40', 'C10', 'C6', 'C37', 'C21',
                   'C31', 'C11', 'C15', 'C34', 'C25', 'C17', 'C4_copy']
multivalue_cols = ['C38', 'C41']

sequence_inputs_len = {}

with open(path) as f:
    for i, line in enumerate(f):
        data = line.strip().split(",")

        for fea in multivalue_cols:
            genres_list = list(reversed(data[HEADER.index(fea)].split('|')))

            if fea not in sequence_inputs_len:
                sequence_inputs_len[fea] = []
            c_l = len(genres_list)
            if len(genres_list) == 1 and str(genres_list[0]) == '-1':
                c_l = 0
            sequence_inputs_len[fea].append(str(c_l))


for fea in multivalue_cols:
    with open("count_"+fea+".txt", "w") as f:
        f.write("\n".join(sequence_inputs_len[fea]))