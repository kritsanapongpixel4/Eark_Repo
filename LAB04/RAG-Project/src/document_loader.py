import os
import zipfile
import xml.etree.ElementTree as ET

def parse_xlsx_rows(file_path):
    """
    Parse .xlsx file using standard library zipfile & xml.etree.
    """
    try:
        z = zipfile.ZipFile(file_path)
        tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        
        rows = []
        for r in tree.findall('.//s:row', ns):
            row_vals = []
            for c in r.findall('s:c', ns):
                is_inline = c.find('s:is', ns)
                v = c.find('s:v', ns)
                if is_inline is not None:
                    val = ''.join([n.text for n in is_inline.iter() if n.text])
                elif v is not None and v.text is not None:
                    val = v.text
                else:
                    val = ''
                row_vals.append(val)
            rows.append(row_vals)
        return rows
    except Exception as e:
        print(f"Error reading XLSX {file_path}: {e}")
        return []

def load_xlsx(file_path):
    rows = parse_xlsx_rows(file_path)
    if not rows or len(rows) < 2:
        return []
    
    headers = rows[0]
    records = []
    
    for idx, row in enumerate(rows[1:], start=1):
        if not any(row):
            continue
        row_dict = dict(zip(headers, row))
        brand = row_dict.get('Brand', '').strip()
        model = row_dict.get('Model Name', '').strip()
        body = row_dict.get('Body Type', '').strip()
        segment = row_dict.get('Segment', '').strip()
        powertrain = row_dict.get('Powertrain Type', '').strip()
        fuel = row_dict.get('Fuel Type', '').strip()
        hp = row_dict.get('Horsepower (HP)', '').strip()
        torque = row_dict.get('Torque (Nm)', '').strip()
        accel = row_dict.get('0100 km/h (s)', row_dict.get('0-100 km/h (s)', '')).strip()
        top_speed = row_dict.get('Top Speed (km/h)', '').strip()
        towing = row_dict.get('Towing Capacity (kg)', '').strip()
        battery = row_dict.get('Usable Battery (kWh)', '').strip()
        range_km = row_dict.get('Real Range (km)', '').strip()
        efficiency = row_dict.get('Efficiency (Wh/km)', '').strip()
        price = row_dict.get('Price (EUR)', '').strip()
        seating = row_dict.get('Seating Capacity', '').strip()
        boot = row_dict.get('Boot Capacity (L)', '').strip()
        adas = row_dict.get('ADAS Level', '').strip()
        ncap = row_dict.get('Safety Rating (Euro NCAP)', '').strip()
        
        question = f"What are the specifications of {brand} {model}?"
        answer = (
            f"The {brand} {model} is a Segment {segment} {body} featuring a {powertrain} powertrain ({fuel}). "
            f"Key Specs: Output {hp} HP and {torque} Nm torque, 0-100 km/h in {accel}s, top speed {top_speed} km/h, "
            f"towing capacity {towing} kg, usable battery {battery} kWh, real driving range {range_km} km, "
            f"efficiency {efficiency} Wh/km, base price €{price}, seating capacity {seating}, boot volume {boot} Liters, "
            f"Euro NCAP safety rating {ncap} stars, and ADAS autonomy Level {adas}."
        )
        
        records.append({
            "id": len(records),
            "category": f"Dataset Specs - {brand}",
            "question": question,
            "answer": answer,
            "source_file": os.path.basename(file_path),
            "line_no": idx
        })
    return records

def load_eu_jp_cars(lines, filename):
    records = []
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("=") or line.startswith("ID") or line.startswith("-") or line.startswith("Note:"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 9 and parts[0].isdigit():
            car_id, make, model, region, country, body, powertrain, power, price = parts[:9]
            question = f"What are the specifications of {make} {model}?"
            answer = (
                f"The {make} {model} (Entry ID {car_id}) is a {body} manufactured in {country} ({region}). "
                f"It is powered by a {powertrain} engine producing {power} with an estimated starting base price of {price} USD."
            )
            records.append({
                "id": len(records),
                "category": f"European & Japanese Cars - {make}",
                "question": question,
                "answer": answer,
                "source_file": filename,
                "line_no": line_no
            })
    return records

def load_glossary_kb(lines, filename):
    records = []
    current_section = "General Automotive Concepts"
    current_topic = ""
    topic_lines = []
    start_line = 1

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("="):
            continue
        if line.startswith("SECTION") or line.startswith("--- SECTION"):
            current_section = line.strip("- =").strip()
            continue
        
        # Topic headers like "1.1 Sedan", "2.1 ICE", "8.1 Level 1 Charging", etc.
        is_topic_header = (
            len(line) > 3 and line[0].isdigit() and "." in line[:4] and (" " in line[:6] or "-" in line[:6])
        )
        
        if is_topic_header:
            if topic_lines and current_topic:
                combined_text = "\n".join(topic_lines)
                records.append({
                    "id": len(records),
                    "category": current_section,
                    "question": f"What is {current_topic}?",
                    "answer": combined_text,
                    "source_file": filename,
                    "line_no": start_line
                })
            current_topic = line
            topic_lines = [line]
            start_line = line_no
        else:
            if current_topic:
                topic_lines.append(line)
            else:
                topic_lines.append(line)

    if topic_lines and current_topic:
        combined_text = "\n".join(topic_lines)
        records.append({
            "id": len(records),
            "category": current_section,
            "question": f"What is {current_topic}?",
            "answer": combined_text,
            "source_file": filename,
            "line_no": start_line
        })
    elif topic_lines:
        combined_text = "\n".join(topic_lines)
        records.append({
            "id": len(records),
            "category": current_section,
            "question": f"General Automotive Knowledge: {current_section}",
            "answer": combined_text,
            "source_file": filename,
            "line_no": 1
        })

    return records

def load_qa_file(file_path):
    """
    Main loader entry point.
    """
    if isinstance(file_path, (list, tuple)):
        records = []
        for path in file_path:
            records.extend(load_qa_file(path))
        for idx, rec in enumerate(records):
            rec["id"] = idx
        return records

    filename = os.path.basename(file_path)

    if file_path.endswith(".xlsx"):
        return load_xlsx(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if "eu_and_jp_cars_dataset.txt" in filename:
        return load_eu_jp_cars(lines, filename)
    elif "car_fundamentals_glossary_kb.txt" in filename:
        return load_glossary_kb(lines, filename)
    else:
        # Generic text loader
        records = []
        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if line and not line.startswith("="):
                records.append({
                    "id": len(records),
                    "category": "General Knowledge",
                    "question": f"Automotive information from {filename}",
                    "answer": line,
                    "source_file": filename,
                    "line_no": line_no
                })
        return records
