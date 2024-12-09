<template>
  <h1 class="heading">Hệ thống phát hiện bệnh trên lá lúa</h1>

  <div class="container">
    <div class="row">
      <div class="col-7">
        <div class="side-left">
          <button type="button" class="btn btn-primary leaf" @click="clickToInputFile(0)">
            Upload hình ảnh cận
          </button>
          <input type="file" id="upload-img" accept="image/png, image/jpeg" hidden @change="displayImg()" />

          <button type="button" style="margin-left: 10px;" class="btn btn-primary field" @click="clickToInputFile(1)">
            Upload hình ảnh cánh đồng
          </button>
          <input type="file" id="upload-field" hidden accept="image/png, image/jpeg" @change="displayImg()" />

          <button type="button" class="btn btn-success" style="margin-left: 10px;" @click="predictClassName()">Nhận
            dạng</button>

          <div>
            <p style="font-style: italic; margin-top: 8px;">
              * Người dùng hãy chọn một hình ảnh cận hoặc hình ảnh đồng lúa để nhận dạng
            </p>
          </div>

          <div class="img-wrapper">
            <img class="cur-img" src="../assets/logo-ctu.png" />
          </div>
        </div>
      </div>
      <div class="col-5">
        <h2>Kết quả nhận dạng: </h2>
        <div class="img-wrapper">
          <img v-if="imageDetect.length > 0" class="detected-img" :src="'data:image/jpeg;base64,' + imageDetect"
            :style="{ width: mode === 1 ? '100%' : '50%' }" />
        </div>
        <br />
        <div v-if="names.length < 5">
          <h3 v-for="(name, index) in names" :key="index">{{ name }} ({{ confidences[index] }})</h3>
        </div>

        <div v-if="classesInfo && classesInfo.length > 0">
          <h5 v-for="(area, j) in classesInfo" :key="j">
            <div v-if="area.area_percentage > 0">
              {{ area.class_name }} chiếm {{ area.area_percentage.toFixed(2) }}% - {{ area.count }} vùng nhiễm bệnh
            </div>
          </h5>
        </div>

        <div v-if="isDetect == 0">
          <p><strong>Hệ thống không phát hiện được bệnh trên cánh đồng</strong></p>
        </div>
      </div>
    </div>

    <div class="row" v-if="currentDiseases.length > 0">
      <div class="col-12">
        <table class="table" border="1">
          <thead>
            <tr>
              <th style="text-align: center;">Tên bệnh</th>
              <th style="text-align: center;">Màu sắc</th>
              <th style="text-align: center;">Gợi ý phòng trừ</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(disease, index) in diseaseData" :key="disease.id">
              <td style="text-align: center;">{{ disease.name }}</td>
              <td style="text-align: center;">{{ disease.color }}</td>
              <td>
                <p v-for="(sug, idx) in disease.suggest" :key="idx">{{ sug }}</p>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
<script>
import axios from 'axios';

export default {
  data() {
    return {
      names: [],
      confidences: [],
      imageDetect: '',
      mode: 0,
      boundingBoxesInfo: [],
      classesInfo: [],
      diseaseData: [],
      currentDiseases: [],
      isDetect: 1
    }
  },
  methods: {
    async clickToInputFile(cls) {
      this.mode = cls
      if (this.mode == 0) {
        const file = document.getElementById('upload-img');
        file.click();
      } else {
        const file = document.getElementById('upload-field');
        file.click();
      }
    },

    async fetchData() {
      try {
        // Gửi request để lấy dữ liệu
        const response = await axios.get('/data.json');

        // Lưu dữ liệu vào biến
        this.diseaseData = response.data;
      } catch (error) {
        console.error('Lỗi khi tải dữ liệu:', error);
      }
    },

    async displayImg() {
      const file = this.mode === 0
        ? document.getElementById('upload-img').files[0]
        : document.getElementById('upload-field').files[0];

      const imgElement = await document.getElementsByClassName('cur-img')[0]
      imgElement.src = URL.createObjectURL(file)
    },

    async predictClassName() {
      this.names = [];
      const classnames = ["Cháy lá", "Đốm nâu", "Đạo ôn"];

      const file = this.mode === 0 ? document.getElementById('upload-img').files[0] : document.getElementById('upload-field').files[0];
      if (!file) {
        console.error("No file selected");
        return;
      }

      const form = new FormData();
      form.append('file', file);

      try {
        const endpoint = this.mode === 0 ? 'http://127.0.0.1:2409/api/predict' : 'http://127.0.0.1:2409/api/predict-field';
        const res = await axios.post(endpoint, form, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        this.currentDiseases = []
        this.classesInfo = []

        const temp = await res.data?.detected_classes;
        this.confidences = await res.data?.confidences;
        
        this.isDetect = await res.data?.isDetect;
        this.imageDetect = await res.data?.detected_image;
        this.boundingBoxesInfo = await res.data?.bounding_boxes_info;
        this.classesInfo = await res.data?.classes_info;

        if(this.classesInfo !== undefined) {
          for (let i = 0; i < this.classesInfo.length; i++) {
            if(this.classesInfo[i].area_percentage > 0) {
              const idx = classnames.indexOf(this.classesInfo[i].class_name)
              this.currentDiseases.push(idx)
            }
          }
        }

        // Map class indexes to names
        if (temp !== undefined) {
          for (let i = 0; i < temp.length; i++) {
            const idx = temp[i];
            this.names.push(classnames[idx]);
          }
          const uniqueTemp = [...new Set(temp)];
          this.currentDiseases = uniqueTemp
          console.log(this.currentDiseases);
          
        }
        
        await this.fetchData();
        
        this.currentDiseases = this.currentDiseases.map(item => String(item));
        
        this.diseaseData = this.diseaseData.filter(disease => this.currentDiseases.includes(String(disease.id)));
      } catch (error) {
        console.error("Error during prediction:", error);
      }
    }
  },
  async created() {
    await this.fetchData(); // Lấy dữ liệu khi component được mount
   
  }
}
</script>
<style scoped>
.heading {
  margin-top: 24px;
  text-align: center;
  color: red;
  text-transform: uppercase;
}

.container {
  margin-top: 32px;
}

.img-wrapper {
  margin-top: 24px;
  width: 100%;
}

.cur-img,
.detected-img {
  width: 50%;
}

.img {
  width: 100%;
}

.btn-recognize {
  display: flex;
  align-items: center;
  height: 100%;
}

.btn-recognize .btn {
  min-width: 100%;
  padding-top: 10px;
  padding-bottom: 10px;
  font-size: 18px;
}

.table {
  margin-top: 35px;
  width: 100%
}
</style>