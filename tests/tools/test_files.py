from plan_execute_agent.tools.files import read_file, write_file, list_dir


def test_write_then_read_file(tmp_path):
    file_path = tmp_path / "note.txt"

    write_result = write_file(str(file_path), "hello world")
    read_result = read_file(str(file_path))

    assert "11 characters" in write_result
    assert read_result == "hello world"


def test_list_dir_shows_files_and_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "subdir").mkdir()

    result = list_dir(str(tmp_path))

    assert "a.txt" in result
    assert "subdir/" in result


def test_list_dir_empty_directory(tmp_path):
    result = list_dir(str(tmp_path))

    assert result == "(empty directory)"
