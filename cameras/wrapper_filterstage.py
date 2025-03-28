from thorlabs_elliptec import ELLx


class FilterStage(object):
    def __init__(self):
        self._filter_stage = ELLx("COM3")

        self.position_list = {'bp680': 0,
                              'lp700': 31,
                              'sp700': 62,
                              'clear': 93}

        self.curr_pos = 'clear'
        self._filter_stage.move_absolute_raw(self.position_list[self.curr_pos], blocking=True)
    
    def set_filter_position(self, position_name: str, blocking=False):
        position_name = position_name.lower()
        if position_name not in self.position_list:
            raise ValueError(f"Invalid filter position name: {position_name}")
        self.curr_pos = position_name
        self._filter_stage.move_absolute_raw(self.position_list[self.curr_pos], blocking=blocking)

    def get_filter_position(self) -> str:
        return self.curr_pos, self.position_list[self.curr_pos]

    def close(self):
        self._filter_stage.close()
    
    def __del__(self):
        self.close()


if __name__ == "__main__":
        
        filter_stage = FilterStage()
        
        while True:
            position_name = input("Enter filter position (None to exit): ")
            if position_name.lower() == 'none':
                filter_stage.close()
                break
            try:
                filter_stage.set_filter_position(position_name, blocking=True)
            except ValueError as e:
                print(e)
                continue
            print(f"Filter position set to {position_name}\r", end="")