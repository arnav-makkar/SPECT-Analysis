#!/usr/bin/env python3
import xml.etree.ElementTree as ET

def test_xml_parsing():
    xml_file = 'control/metadata/PPMI_3000_Reconstructed_DaTSCAN_S117534_I323662.xml'
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        print(f"Root tag: {root.tag}")
        print(f"Direct children: {[child.tag for child in root]}")
        
        # Try different namespace approaches
        ns = {'ida': 'http://ida.loni.usc.edu'}
        
        # Method 1: Using namespace dictionary
        subject = root.find('.//ida:subject', ns)
        print(f"Subject found (ns dict): {subject is not None}")
        
        if subject:
            subject_id = subject.find('ida:subjectIdentifier', ns)
            if subject_id is not None:
                print(f"Subject ID: {subject_id.text}")
            else:
                print("Subject ID not found")
        
        # Method 2: Using full namespace in path
        subject2 = root.find('.//{http://ida.loni.usc.edu}subject')
        print(f"Subject found (full ns): {subject2 is not None}")
        
        if subject2:
            subject_id2 = subject2.find('.//{http://ida.loni.usc.edu}subjectIdentifier')
            if subject_id2 is not None:
                print(f"Subject ID 2: {subject_id2.text}")
            else:
                print("Subject ID 2 not found")
        
        # Method 3: Search without namespace
        subject3 = root.find('.//subject')
        print(f"Subject found (no ns): {subject3 is not None}")
        
        if subject3:
            subject_id3 = subject3.find('.//subjectIdentifier')
            if subject_id3 is not None:
                print(f"Subject ID 3: {subject_id3.text}")
            else:
                print("Subject ID 3 not found")
        
        # Print all elements to see structure
        print("\nAll elements found:")
        for elem in root.iter():
            print(f"  {elem.tag}: {elem.text if elem.text and elem.text.strip() else 'No text'}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_xml_parsing()
