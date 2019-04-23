from excel_class import XL, Printer
class DataLoader():
    def __init__(self, frd):
        self.frd = frd
        self.meta = []
        self.data_tr = []
        self.data_tst = []
        self.label_tr = []
        self.label_tst = []
        # use control / patient groups . false means control / p_right / p_left groups
        # self.gr_2 = False
        self.gr_2 = True

    def data_read(self):
        printer = Printer()
        xl_file_name = '/home/sp/fsl/T1_miguel/subcortical_volume.xlsx'
        xl = XL(xl_file_name, 'read')
        ws_list = xl.get_sh_names()
        data = []
        for i in range(3):
            xl.open_ws(ws_list[i])
            data.append(xl.rd_all_row())
            del data[i][0]

        print(len(data))
        #printer.p_list(data)
        return data

    def extr_data(self, data):
        for i in range(3):
            self.meta.append(self.extr_gr(data[i],i))

    def get_gr_name(self, gr_num):
        if gr_num == 0:
            gr = 'control'
        elif gr_num == 1:
            gr = 'p_left'
        elif gr_num == 2:
            gr = 'p_right'
        return gr

    def extr_gr(self, group, gr_num):
        gr_data = []
        gr = self.get_gr_name(gr_num)
        for hum in group:
            #print(hum)
            print(gr)
            datum = Datum(gr)
            datum.set_info(hum)
            gr_data.append(datum)
        return gr_data

    def get_tr_set(self):
        for i in range(3):
            #print(self.meta[i])
            self.get_tr_gr(self.meta[i], i, self.frd)
        print('training set num in group : {}'.format(len(self.data_tr)))
        return self.data_tr, self.label_tr

    def get_tr_gr(self, gr, gr_num, frd):
        cnt = 0
        if self.gr_2 and gr_num == 2: gr_num = 1
        for hum in gr:
            if self.is_tr(cnt, frd):
                #hum.print()
                self.data_tr.append(hum.get_data())
                self.label_tr.append(gr_num)
            cnt = cnt+1

    def get_tst_set(self):
        for i in range(3):
            self.get_tst_gr(self.meta[i], i, self.frd)
        print('testing set num in group : {}'.format(len(self.data_tst)))
        return self.data_tst, self.label_tst

    def get_tst_gr(self, gr, gr_num, frd):
        cnt = 0
        if self.gr_2 and gr_num == 2: gr_num = 1
        for hum in gr:
            #print(hum.get_data())
            if not self.is_tr(cnt, frd):
                self.data_tst.append(hum.get_data())
                self.label_tst.append(gr_num)
            cnt = cnt+1

    def get_gr_by_i(self, index):
        data_l = []
        label_l = []
        for hum in self.meta[index]:
            if hum.get_gr() == self.get_gr_name(index):
                data_l.append(hum.get_data())
                label_l.append(index)
        return data_l, label_l

    def is_tr(self, cnt, frd):
        if cnt%frd != 0:
            return True
        else:
            return False

class Datum():
    def __init__(self, gr):
        self.name = ''
        self.gr = gr
        self.data = []

    def set_info(self, line):
        if len(line) < 2:
            print('invalid datum information : ')
            print(line)
            return
        self.name = line[1][:-1]
        for i in range(2,len(line)):
            self.data.append(int(line[i]))
            #self.data.append([line[i]])

    def print(self):
        print(self.name)
        print(self.gr)
        print(self.data)

    def get_data(self):
        return self.data

    def get_data_col(self):
        pass

    def get_gr(self):
        return self.gr
#prt = Printer()
#loader = DataLoader(2)
#data = loader.data_read()
#loader.extr_data(data)
#data_tr, label_tr = loader.get_tr_set()
#data_tst, label_tst = loader.get_tst_set()
#prt.p_list(data_tr)
#prt.p_list(label_tr)

