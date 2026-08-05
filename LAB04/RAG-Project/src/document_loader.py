import os
import zipfile
import xml.etree.ElementTree as ET

def parse_xlsx_rows(file_path):
    """
    Parse .xlsx file using standard library zipfile & xml.etree
    without requiring openpyxl.
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
        brand = row_dict.get('Brand', '')
        model = row_dict.get('Model Name', '')
        body = row_dict.get('Body Type', '')
        powertrain = row_dict.get('Powertrain Type', '')
        fuel = row_dict.get('Fuel Type', '')
        hp = row_dict.get('Horsepower (HP)', '')
        price = row_dict.get('Price (EUR)', '')
        range_km = row_dict.get('Real Range (km)', '')
        ncap = row_dict.get('Safety Rating (Euro NCAP)', '')
        
        question = f"What are the specifications of {brand} {model}?"
        answer = f"The {brand} {model} is a {body} featuring a {powertrain} powertrain ({fuel}). It delivers {hp} HP, has a real-world driving range of {range_km} km, a base price of €{price}, and a Euro NCAP safety rating of {ncap} stars."
        
        records.append({
            "id": len(records),
            "category": f"Vehicle Dataset - {brand}",
            "question": question,
            "answer": answer,
            "source_file": os.path.basename(file_path),
            "line_no": idx
        })
    return records

def load_txt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    current_category = "Automotive Knowledge"
    pending_question = None
    pending_line_no = None
    filename = os.path.basename(file_path)

    # If it's a Q&A question file with Q: and A:
    has_qa_lines = any(line.strip().startswith(("Q:", "A:", "Q0", "Q1")) for line in lines)

    if has_qa_lines:
        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if line.startswith("#") or line == "" or line.startswith("="):
                continue
            if line.startswith("--- SECTION"):
                current_category = line.strip("- ").strip()
                continue
            if line.startswith("Q:") or (line.startswith("Q") and ":" in line[:6]):
                q_text = line.split(":", 1)[1].strip()
                records.append({
                    "id": len(records),
                    "category": current_category,
                    "question": q_text,
                    "answer": f"Question inquiry for RAG index: {q_text}",
                    "source_file": filename,
                    "line_no": line_no
                })
        return records

    # General structured text file (like car_fundamentals_glossary_kb.txt or eu_and_jp_cars_dataset.txt)
    current_section = "General Automotive Concepts"
    block_lines = []

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if line.startswith("="):
            continue
        if line.startswith("SECTION") or line.startswith("--- SECTION"):
            current_section = line.strip("- =").strip()
            continue
        if line:
            block_lines.append((line_no, line))
            if len(block_lines) >= 4:
                combined_text = " ".join([b[1] for b in block_lines])
                records.append({
                    "id": len(records),
                    "category": current_section,
                    "question": f"Automotive knowledge regarding {current_section}",
                    "answer": combined_text,
                    "source_file": filename,
                    "line_no": block_lines[0][0]
                })
                block_lines = []

    if block_lines:
        combined_text = " ".join([b[1] for b in block_lines])
        records.append({
            "id": len(records),
            "category": current_section,
            "question": f"Automotive knowledge regarding {current_section}",
            "answer": combined_text,
            "source_file": filename,
            "line_no": block_lines[0][0]
        })

    return records

def load_qa_file(file_path):
    """
    Main loader entry point supporting list of files, .txt, and .xlsx files.
    """
    if isinstance(file_path, (list, tuple)):
        records = []
        for path in file_path:
            records.extend(load_qa_file(path))
        # Re-index IDs sequentially
        for idx, rec in enumerate(records):
            rec["id"] = idx
        return records

    if file_path.endswith(".xlsx"):
        return load_xlsx(file_path)
    else:
        return load_txt(file_path)
