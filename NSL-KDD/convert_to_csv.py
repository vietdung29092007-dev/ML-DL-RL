"""Convert NSL-KDD .txt files to .csv with proper headers."""
import os

header = "duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate,label,difficulty_level"

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

for name in ["KDDTrain+", "KDDTest+"]:
    input_file = os.path.join(script_dir, f"{name}.txt")
    output_file = os.path.join(script_dir, f"{name}.csv")
    
    try:
        with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
            f_out.write(header + "\n")
            for line in f_in:
                f_out.write(line)
        print(f"Created {name}.csv")
    except FileNotFoundError as e:
        print(f"Error: Could not find {name}.txt - {e}")
    except Exception as e:
        print(f"Error processing {name}: {e}")

print("Done!")
