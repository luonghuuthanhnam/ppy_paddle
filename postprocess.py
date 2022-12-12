import pandas as pd
import re
import unidecode
import numpy as np

class KiePostProcess():
    def __init__(self):
        self.predicted_kie_df = None
        self.PATIENT_NAME_REMOVE_LIST=[
            "Họ và tên người bệnh",
            "họ và tên người bệnh",
            "Họ và tên",
            "họ và tên",
            "Họ tên",
            "họ tên",
            "Bệnh nhân",
            "bệnh nhân",
        ]
        self.AGE_ANCHOR_LIST = [
        "tuoi",
        "nam sinh",
        "ngay sinh",
        "sinh ngay",
        "sinh nam",
        "nam sinh"
        ]

        self.GENDER_ANCHOR_LIST = [
            "gioi tinh",
            "gioi",
            "nam/nu",
            "nam/ nu",
            "nam / nu",
            "nam /nữ",
        ]

        self.LABEL_LIST = ["None", "sign_date", "diagnose", "hospital_name", "address", "age", "treatment", "patient_name", "admission_date", "discharge_date", "gender", "document_type", "department", "note", "BHYT",]

    def append_kie_df(self, predicted_kie_df):
        self.predicted_kie_df = predicted_kie_df

    def get_raw_predicted(self, label:str, predicted_df:pd.DataFrame, sort_top_down = False, return_xy =False) -> list: 
        extracted_row = predicted_df[predicted_df["pred_label"]==label]
        if sort_top_down == True:
            extracted_row = extracted_row.sort_values("ymax", ascending=True)
        else:
            extracted_row = extracted_row.sort_values("confidence_score", ascending=False)
        extracted_text = extracted_row["Object"].to_list()
        xmax = extracted_row["xmax"].to_list()
        xmin = extracted_row["xmin"].to_list()
        ymax = extracted_row["ymax"].to_list()
        ymin = extracted_row["ymin"].to_list()
        gcn_scores = extracted_row["confidence_score"].to_list()
        extracted_text = [each.strip() for each in extracted_text]
        if return_xy == True:
            return extracted_text, gcn_scores, (xmax, xmin, ymax, ymin)
        else:
            return extracted_text, gcn_scores

    def patient_name_regularization(self, raw_patient_name):
        shorten_patient_name = raw_patient_name
        count_spliter = raw_patient_name.count(":")
        if count_spliter>0:
            shorten_patient_name = raw_patient_name.split(":")[-1]
        for each in self.PATIENT_NAME_REMOVE_LIST:
            shorten_patient_name = shorten_patient_name.replace(each, "")
        return shorten_patient_name.strip()

    def parse_date_with_re(self, raw_date):
        pattern = "\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        temp_date = raw_date.replace("\\","/")
        temp_date = raw_date.replace(" ","")
        temp_date = re.findall(pattern=pattern, string=temp_date)
        list_result = []
        for each in temp_date:
            for splitter in ["/", "-"]:
                if splitter in each:
                    d, m, y = each.split(splitter)
                    list_result.append((d,m,y))
                    break
        return list_result

    def vn_date_parser(self, date_raw_text):
        unaccented_string = unidecode.unidecode(date_raw_text).lower()
        day_idx = -1
        day_val = -1
        month_idx = -1
        month_val = -1
        year_idx = -1 
        year_val = -1
        
        list_dates = self.parse_date_with_re(unaccented_string)
        if len(list_dates) ==0:
            unaccented_string = re.sub('(\d+(\.\d+)?)', r'\1 ', unaccented_string) #append space between mixed digit and charactor
            unaccented_string = re.sub(' +', ' ', unaccented_string)
            elms = unaccented_string.split(" ")
            max_idx = len(elms) -1
            
            #Parse Day
            try:
                day_idx = elms.index("ngay") + 1
                temp_day_val = int(elms[day_idx])
                if temp_day_val in range(1,32):
                    day_val = temp_day_val
            except Exception as e:
                # print(e)
                # print("***cannot parse DAY***")
                pass

            #Parse Month
            try:
                month_idx = elms.index("thang") + 1
                temp_month_val = int(elms[month_idx])
                if temp_month_val in range(1,13):
                    month_val = temp_month_val
            except Exception as e:
                # print(e)
                # print("***cannot parse Month***")
                pass

            
            #Parse Year
            try:
                year_idx = elms.index("nam") + 1
                temp_year_val = int(elms[year_idx])
                if temp_year_val in range(1900,2100):
                    year_val = temp_year_val
            except Exception as e:
                # print(e)
                # print("***cannot parse Year***")
                pass
        else:
            day_val, month_val, year_val = list_dates[0]
        return day_val, month_val, year_val

    def is_mixed_age_gender(self, raw_text):
        count_spliter = raw_text.count(":")
        is_mixed = False
        if count_spliter>1:
            is_mixed = True
        temp_age_elm = False
        temp_gender_elm = False
        unaccented_checking_text = unidecode.unidecode(raw_text).lower()
        for each in self.AGE_ANCHOR_LIST:
            if each in unaccented_checking_text:
                temp_age_elm = True
                break
        for each in self.GENDER_ANCHOR_LIST:
            if each in unaccented_checking_text:
                temp_gender_elm = True
                break
        if temp_age_elm + temp_gender_elm ==2:
            is_mixed = True
        return is_mixed

    def check_mixed_age_gender(self, raw_text):
        age = None
        gender = None
        parts = raw_text.split(":")
        list_number = re.findall(r'\d+', raw_text)
        if len(list_number)==1:
            age = list_number[0]
            unit_flag = sum([True if each in raw_text.lower() else False for each in ["th", "tháng"]])
            if unit_flag != 0:
                age = int(age)/12
        
        raw_text_check = raw_text.replace(" ", "").lower()
        for each in ["xnam", "xnữ", "namx" ,"nữx"]:
            if gender != None:
                break
            else:
                if each in raw_text_check:
                    if "nam" in each:
                        gender = "Nam"
                        break
                    else:
                        gender = "Nữ"
                        break
                    
        if gender == None:
            words = raw_text.split(" ")
            if words[-1].lower().strip() in ["nam"]:
                gender = "Nam"
            elif words[-1].lower().strip() == "nữ":
                gender = "Nữ"
        
        
        if len(list_number) > 1:
            # print("Check birthday")
            try:
                # birthday = dparser.parse(raw_text,fuzzy=True, dayfirst = True)
                d,m,y = self.vn_date_parser(raw_text)
                age = f"{d}/{m}/{y}"
            except:
                pass
                # print("***cannot parse date from:", raw_text)
            #Check birthday
            pass
        return age, gender

    
    def age_regularization(self, raw_age):
        age = None
        gender = None
        count_spliter = raw_age.count(":")
        is_mixed = self.is_mixed_age_gender(raw_age)
        if is_mixed == True:
            temp_age, temp_gender = self.check_mixed_age_gender(raw_age)
            if temp_age != None:
                age = temp_age
            if temp_gender != None:
                gender = temp_gender
        elif is_mixed == False and count_spliter == 1:
            left_part, right_part = raw_age.split(":")
            left_part = left_part.strip()
            right_part = right_part.strip()
            unaccented_checking_text = unidecode.unidecode(left_part).lower()
            for age_anchor in self.AGE_ANCHOR_LIST:
                if age_anchor in unaccented_checking_text:
                    list_number = re.findall(r'\d+', right_part)
                    if len(list_number)==1:
                        age = list_number[0]
                        unit_flag = sum([True if each in right_part.lower() else False for each in ["th", "tháng"]])
                        if unit_flag != 0:
                            age = int(age)/12
                    elif len(list_number) > 1:
                        # print("Check birthday")
                        try:
                            d,m,y = self.vn_date_parser(raw_age)
                            age = f"{d}/{m}/{y}"
                        except:
                            pass
                            # print("***cannot parse date from:", raw_age)
                        #Check birthday
                    break

        elif is_mixed == False and count_spliter == 0:
            list_number = re.findall(r'\d+', raw_age)
            if len(list_number) ==1:
                age = list_number[0]
                unit_flag = sum([True if each in raw_age.lower() else False for each in ["th", "tháng"]])
                if unit_flag != 0:
                    age = int(age)/12
            elif len(list_number) >1:
                #Check birthday
                # print("Check birthday")
                try:
                    d,m,y = self.vn_date_parser(raw_age)
                    age = f"{d}/{m}/{y}"
                except:
                    pass
                    # print("***cannot parse date from:", raw_age)
        return age, gender

    
    def gender_regularization(self, raw_gender):
        age = None
        gender = None
        count_spliter = raw_gender.count(":")

        is_mixed = self.is_mixed_age_gender(raw_gender)

        if is_mixed == True:
            temp_age, temp_gender = self.check_mixed_age_gender(raw_gender)
            if temp_age != None:
                age = temp_age
            if temp_gender != None:
                gender = temp_gender
        
        elif is_mixed == False and count_spliter == 1:
            left_part, right_part = raw_gender.split(":")
            left_part = left_part.strip()
            right_part = right_part.strip()
            unaccented_checking_text = unidecode.unidecode(left_part).lower()
            for each in self.GENDER_ANCHOR_LIST:
                if each in unaccented_checking_text.lower():
                    if right_part.lower() == "nam":
                        gender = "Nam"
                    elif right_part.lower() == "nữ":
                        gender = "Nữ"
                    break

        elif is_mixed == False and count_spliter == 0:
            shorten_gender = raw_gender
            unaccented_checking_text = unidecode.unidecode(shorten_gender).lower()
            for each in self.GENDER_ANCHOR_LIST:
                unaccented_checking_text = unaccented_checking_text.replace(each, "").replace(" ", "")
            
                for each in ["xnam", "xnu", "namx", "nux"]:
                    if gender != None:
                        break
                    else:
                        if each in unaccented_checking_text:
                            if "nam" in each:
                                gender = "Nam"
                                break
                            else:
                                gender = "Nữ"
                                break
                            
                if gender == None:
                    temp_unaccented_checking_text = unaccented_checking_text.strip().split()
                    if temp_unaccented_checking_text[-1] == "nam":
                        gender = "Nam"
                    elif temp_unaccented_checking_text[-1] == "nu":
                        gender = "Nữ"
        return gender, age


    def icd_code_parser(self, raw_diagnose):
        diagnose = " " + raw_diagnose + " "
        diagnose = diagnose.replace("ICD10", "ICD")
        diagnose = diagnose.replace("icd10", "icd")
        first_pattern = '(?=([,:\- \[({;/]([a-zA-Z]{1}\d+.\d+)[*.,:\- \])};/]))'
        second_pattern = '(?=([,:\- \[({;/]([a-zA-Z]{1}\d+)[*.,:\- \])};/]))'
        first_result = re.findall(first_pattern, diagnose)
        second_result = re.findall(second_pattern, diagnose)

        first_result_unique  = [each[1] for each in first_result]
        second_result_unique  = [each[1] for each in second_result]

        first_result_idx = [raw_diagnose.index(each) for each in first_result_unique]
        second_result_idx = [raw_diagnose.index(each) for each in second_result_unique]

        unique_icds = first_result_unique.copy()
        for each_idx, val in zip(second_result_idx, second_result_unique):
            if each_idx not in first_result_idx:
                unique_icds.append(val)
        return unique_icds

    
    def CheckMergeHospitalNamevsORG(self, hospital_name):
        simple_hospital_name = unidecode.unidecode(hospital_name).strip().lower()
        simple_hospital_name = simple_hospital_name.replace(" ", "")
        check_list = ["benhviendakhoatinh", "benhviendktinh", "bvdktinh", "benhviendakhoahuyen", "benhviendkhuyen", "bvdkhuyen",
                      "benhviendakhoathanhpho", "benhviendkthanhpho", "bvdkthanhpho", "benhviendakhoatp", "benhviendakhoaquan",
                      "benhviendakhoakhuvuc", "benhviendakhoa", "trungtamytehuyen", "trungtamyte", "ttythuyen", "ttytehuyen",
                      "benhviensannhi", "benhviennhi", "benhvienphusan", "benhviensannhitinh", "benhvienphusantinh"]
        new_hospital_name = None
        if simple_hospital_name in check_list:
            extracted_org, extracted_scores =  self.get_raw_predicted("org", self.predicted_kie_df, sort_top_down=False, return_xy=False)
            if len(extracted_org)>0:
                extracted_org = extracted_org[0]
            else:
                extracted_org = ""
            simple_org = unidecode.unidecode(extracted_org).lower()
            org_remove_list = ["so y te", "uy ban nhan dan", "UBND"]
            split_point = None
            for each in org_remove_list:
                if each in simple_org:
                    start_point = simple_org.find(each)
                    if start_point != -1:
                        split_point = start_point + len(each)
                    break
            if split_point != None:
                left_part = hospital_name.strip()
                right_part = extracted_org[split_point:].strip()
                left_elms = left_part.split(" ")
                right_elms = right_part.split(" ")
                if left_elms[-1].lower() == right_elms[0].lower() and len(right_elms) > 1:
                    new_hospital_name = " ".join(left_elms) + " " + " ".join(right_elms[1:])
                else:
                    new_hospital_name = left_part + " " + right_part
        if new_hospital_name == None:
            new_hospital_name = hospital_name
        return new_hospital_name

    def remove_false_hospital_name(self, extracted_hospital_name:list):
        true_list = ["benhvien", "dakhoa", "ttyt", "trungtam", "tramyte", "bv", "dk", "cotruyen", "vien"]
        false_list = ["soyte", "khoa", "ubnd", "ms", "conghoaxahoi", "luu tru", "ma so"]
        reduced_extracted_hospital_name = []
        if len(extracted_hospital_name) >0:
            for each in extracted_hospital_name:
                simple_each = unidecode.unidecode(each).lower().replace(" ","")
                added_flag = False
                for true_elm in true_list:
                    if true_elm in simple_each:
                        reduced_extracted_hospital_name.append(each)
                        added_flag = True
                        break
        
                if added_flag==False:
                    for false_elm in false_list:
                        if false_elm in simple_each:
                            added_flag = True
                            break
                if added_flag == False:
                    reduced_extracted_hospital_name.append(each)
        start_idx = 0
        if len(reduced_extracted_hospital_name) >1:
            break_flag = False
            for idx, each in enumerate(reduced_extracted_hospital_name):
                simple_each = unidecode.unidecode(each).lower().replace(" ","")
                for true_elm in true_list:
                    if true_elm in simple_each:
                        start_idx = idx
                        break_flag = True
                        break
                if break_flag == True:
                    break
        reduced_extracted_hospital_name = reduced_extracted_hospital_name[start_idx:]
        return reduced_extracted_hospital_name

    def hospital_name_postprocess(self):
        extracted_hospital_name, extracted_scores, (xmax, xmin, ymax, ymin) =  self.get_raw_predicted("hospital_name", self.predicted_kie_df, sort_top_down=True, return_xy=True)
        clean_extracted_hospital_name = self.remove_false_hospital_name(extracted_hospital_name)
        merged_hospital_name = None
        if len(clean_extracted_hospital_name) >0:
            merged_hospital_name = clean_extracted_hospital_name[0]
            if len(clean_extracted_hospital_name)>1:
                for i in range(1, len(clean_extracted_hospital_name)):
                    cur_cen_x = int((xmax[i] + xmin[i])/2)
                    cur_cen_y = int((ymax[i] + ymin[i])/2)
                    pre_cen_x = int((xmax[i] + xmin[i-1])/2)
                    pre_cen_y = int((ymax[i-1] + ymin[i-1])/2)
                    x_dis = np.abs(cur_cen_x - pre_cen_x)
                    y_dis = np.abs(cur_cen_y - pre_cen_y)
                    if x_dis <= 100 and y_dis <= 50:
                        merged_hospital_name += f" {clean_extracted_hospital_name[i]}"
        mean_score = np.mean(np.array(extracted_scores))
        if merged_hospital_name != None:
            merged_hospital_name = self.CheckMergeHospitalNamevsORG(merged_hospital_name)
        return merged_hospital_name, mean_score

    def patient_family_name_correction(self, full_name):
        CORRECTING_FAMILY_NAME = {
            "nguyen": "nguyễn",
            "tran": "trần",
            "le": "lê",
            "pham": "phạm",
            "phan": "phan",
            "huynh": "huỳnh",
            "hoang": "hoàng",
            "vu": "vũ",
            "vo": "võ",
            "dang": "đặng",
            "bui": "bùi",
            "do": "đỗ",
            "ho": "hồ",
            "ngo": "ngô",
        }
        raw_family_name = full_name.split(" ")[0]
        raw_name_remain = full_name.split(" ")[1:]
        try:
            retrieved_family_name = CORRECTING_FAMILY_NAME[unidecode.unidecode(raw_family_name).lower()]
            corrected_family_name = []
            for char_raw, char_re in zip(raw_family_name, retrieved_family_name):
                save_char = char_re
                if char_raw.isupper():
                    save_char = char_re.upper()
                corrected_family_name.append(save_char)
            corrected_family_name = "".join(corrected_family_name)
        except:
            corrected_family_name = raw_family_name
        
        corrected_name = " ".join([corrected_family_name] + raw_name_remain)
        return corrected_name
        
        
    def patient_name_postprocess(self):
        extracted_patient_names, extraced_scores =  self.get_raw_predicted("patient_name", self.predicted_kie_df)
        patient_name = None
        extraced_score = 0
        if len(extracted_patient_names)>0:
            patient_name = self.patient_name_regularization(extracted_patient_names[0])
            extraced_score = extraced_scores[0]
        return patient_name, extraced_score

    def age_postprocess(self):
        extracted_ages, extraced_scores =  self.get_raw_predicted("age", self.predicted_kie_df)
        revert_extracted_ages = extracted_ages.copy()
        revert_extraced_scores = extraced_scores.copy()
        revert_extracted_ages.reverse()
        revert_extraced_scores.reverse()
        age, gender = None, None
        age_score, gender_score = 0,0
        # if len(extracted_ages)>0:
        for each, score in zip(revert_extracted_ages, revert_extraced_scores):
            temp_age, temp_gender = self.age_regularization(each)
            if temp_age!= None:
                age = temp_age
                age_score = score
            if temp_gender!= None:
                gender = temp_gender
                gender_score = score
        return (age, age_score), (gender, gender_score)

    def gender_postprocess(self):
        extracted_gender, extracted_scores =  self.get_raw_predicted("gender", self.predicted_kie_df)
        gender, age = None, None
        age_score, gender_score = 0,0
        if len(extracted_gender)>0:
            gender, temp_age = self.gender_regularization(extracted_gender[0])
            score = extracted_scores[0]
            if temp_age!= None:
                age = temp_age
                age_score = score
            if gender != None:
                gender_score = score
        return (gender, gender_score), (age, age_score)

    def admission_date_postprocess(self):
        extracted_admission_date, extracted_scores =  self.get_raw_predicted("admission_date", self.predicted_kie_df)
        admission_dates = []
        admission_dates_scores = []
        if len(extracted_admission_date)>0:
            for each, score in zip(extracted_admission_date, extracted_scores):
                d,m,y = self.vn_date_parser(each)
                if d!= -1 and m!= -1 and y!= -1:
                    vis_date = f"{d}/{m}/{y}"
                    admission_dates.append(vis_date)
                    admission_dates_scores.append(score)
        return admission_dates, admission_dates_scores


    def discharge_date_postprocess(self):
        extracted_discharge_date, extracted_scores =  self.get_raw_predicted("discharge_date", self.predicted_kie_df)
        discharge_dates = []
        discharge_dates_scores = []
        if len(extracted_discharge_date)>0:
            for each, score in zip(extracted_discharge_date, extracted_scores):
                d,m,y = self.vn_date_parser(each)
                if d!= -1 and m!= -1 and y!= -1:
                    vis_date = f"{d}/{m}/{y}"
                    discharge_dates.append(vis_date)
                    discharge_dates_scores.append(score)
        return discharge_dates, discharge_dates_scores

    def sign_date_postprocess(self):
        extracted_sign_date, extracted_scores =  self.get_raw_predicted("sign_date", self.predicted_kie_df)
        sign_dates = []
        sign_dates_scores = []
        if len(extracted_sign_date)>0:
            for each, score in zip(extracted_sign_date, extracted_scores):
                d,m,y = self.vn_date_parser(each)
                if d!= -1 and m!= -1 and y!= -1:
                    vis_date = f"{d}/{m}/{y}"
                    sign_dates.append(vis_date)
                    sign_dates_scores.append(score)
        return sign_dates, sign_dates_scores

    def ICD_code_postprocess(self):
        extracted_ICD_code, extracted_ICD_code_scores = self.get_raw_predicted("diagnose", self.predicted_kie_df)
        unique_icd_codes = []
        if len(extracted_ICD_code)>0:
            for each in extracted_ICD_code:
                icd_code = self.icd_code_parser(each)
                unique_icd_codes += icd_code
        remain_field =  self.get_raw_predicted("None", self.predicted_kie_df)[0] + self.get_raw_predicted("treatment", self.predicted_kie_df)[0] + self.get_raw_predicted("note", self.predicted_kie_df)[0]
        for each in remain_field:
            if "icd" in each.lower():
                icd_code = self.icd_code_parser(each)
                unique_icd_codes += icd_code
        unique_icd_codes = list(set(unique_icd_codes))
        return unique_icd_codes

    def find_age_remain(self):
        check_list = ["tuổi", "ngày sinh", "năm sinh"]
        age, temp_gender = None, None
        age_score, temp_gender_score = 0,0
        remain_field = []
        remain_field_scores = []
        for cls in self.LABEL_LIST:
            fieds, scores = self.get_raw_predicted(cls, self.predicted_kie_df)
            remain_field +=  fieds
            remain_field_scores +=  scores

        for each, score in zip(remain_field, remain_field_scores):
            for i in check_list:
                if i in each.lower():
                    splited_text = each.lower().split(i)[-1]
                    age, temp_gender = self.age_regularization(splited_text)
                    if age!= None:
                        age_score = score
                    if temp_gender != None:
                        temp_gender_score = score
                    break
            if age!= None:
                break
        return (age, age_score), (temp_gender, temp_gender_score)

    def find_patient_name_remain(self):
        name = None
        name_score = 0
        # remain_field = []
        # for cls in self.LABEL_LIST:
        #     remain_field +=  self.get_raw_predicted(cls, self.predicted_kie_df)
        remain_field, extracted_scores = self.get_raw_predicted("document_type", self.predicted_kie_df)

        for each, score in zip(remain_field, extracted_scores):
            list_exception = ["giayravien", "hoadonvienphi", "hoadonthuphi", "donthuoc", "giaychungnhanphauthuat", "bangke"]
            upper_each = each.strip().upper()
            unaccented_lower_each = unidecode.unidecode(upper_each).lower().replace(" ", "")
            
            if each == upper_each and unaccented_lower_each not in list_exception:
                if len(each.split()) in range(1, 8):
                    name = each
                    name_score = score
            if name!= None:
                break
        return name, name_score

    def find_hospital_name_remain(self):
        check_list = ["benhvien", "trungtamyte", "ttyt", "bvdk", "benhxa"]
        hospital_name = None
        hospital_name_score = 0
        # for cls in self.LABEL_LIST:
        remain_field, extracted_score =  self.get_raw_predicted("department", self.predicted_kie_df)
        for each, score in zip(remain_field, extracted_score):
            unaccented_checking_text = unidecode.unidecode(each).lower().replace(" ","")
            for i in check_list:
                if i in unaccented_checking_text:
                    hospital_name = each
                    hospital_name_score = score
                    break
            if hospital_name != None:
                break
        return hospital_name, hospital_name_score


    def admission_discharge_correction(self, admission_dates, admission_dates_scores, discharge_dates, discharge_dates_scores):
        total_dates = []
        digit_total_dates = []
        total_dates_scores = []
        admission_dates_out = None
        discharge_dates_out = None
        
        admission_dates_scores_out = None
        discharge_dates_scores_out = None
        
        temp_total_dates = admission_dates.copy() + discharge_dates.copy()
        temp_total_dates_scores = admission_dates_scores.copy() + discharge_dates_scores.copy()
        for each, score in zip(temp_total_dates, temp_total_dates_scores):
            if each not in total_dates:
                d,m,y = each.split("/")
                d = "{:02d}".format(int(d))
                m = "{:02d}".format(int(m))
                y = "{:04d}".format(int(y))

                digit_date = int(f"{y}{m}{d}")
                digit_total_dates.append(digit_date)
                new_string_date = f"{d}/{m}/{y}"
                total_dates.append(new_string_date)
                total_dates_scores.append(score)
        if len(total_dates) > 1:
            zipped = zip(total_dates, digit_total_dates, total_dates_scores)
            sorted_dates = sorted(zipped, key = lambda x: x[1])
            admission_dates_out = sorted_dates[0][0]
            admission_dates_scores_out = sorted_dates[0][2]
            
            discharge_dates_out = sorted_dates[-1][0]
            discharge_dates_scores_out = sorted_dates[-1][2]
            
        elif len(total_dates) == 1:
            if admission_dates != []:
                d,m,y = admission_dates[0].split("/")
                d = "{:02d}".format(int(d))
                m = "{:02d}".format(int(m))
                y = "{:04d}".format(int(y))
                new_string_date = f"{d}/{m}/{y}"
                admission_dates_out = new_string_date
                admission_dates_scores_out = admission_dates_scores
            else:
                d,m,y = discharge_dates[0].split("/")
                d = "{:02d}".format(int(d))
                m = "{:02d}".format(int(m))
                y = "{:04d}".format(int(y))
                new_string_date = f"{d}/{m}/{y}"
                discharge_dates_out = new_string_date
                discharge_dates_scores_out = discharge_dates_scores
        return admission_dates_out, admission_dates_scores_out, discharge_dates_out, discharge_dates_scores_out
            
