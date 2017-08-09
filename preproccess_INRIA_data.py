import re

with open('INRIAPerson/Train/pos.lst') as pos_list, \
     open('pr_pos.txt', 'w') as new_pos, \
     open('INRIAPerson/Train/annotations.lst', 'r') as annotations_list:

    for sample_path, annotation_path in zip(pos_list, annotations_list):

        with open('INRIAPerson/' + annotation_path[:-1], 'r', encoding='ISO-8859-15') as annotation_description:
            counter = 0
            coordinates = list()
            for line in annotation_description:
                if line.startswith('Bounding'):
                    x_min, y_min, x_max, y_max = [int(el) for el in re.findall(r'(?<=\()\d+|\d+(?=\))', line)]
                    width = x_max - x_min
                    height = y_max - y_min
                    values = map(str, (x_min, y_min, width, height))
                    counter += 1
                    coordinates.append(values)

            new_line = 'INRIAPerson/{image_path} {count_objects} '.format(image_path=sample_path[:-1], count_objects=counter)
            for obj_val in coordinates:
                new_line += ' '.join(obj_val) + ' '
            new_line += '\n'
            new_pos.write(new_line)