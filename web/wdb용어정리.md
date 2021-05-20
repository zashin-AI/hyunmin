## form 태그
form 태그의 enctype 속성은 폼 데이터(form data)가 서버로 제출될 때 해당 데이터가 인코딩되는 방법을 명시합니다.

* enctype="속성값"
    - multipart/form-data : 모든 문자를 인코딩하지 않음을 명시함. 이 방식은 'form' 요소가 파일이나 이미지를 서버로 전송할 때 주로 사용함.
    - application/x-www-form-urlencoded : 기본값으로, 모든 문자들은 서버로 보내기 전에 인코딩됨을 명시함.
    - text/plain : 공백 문자(space)는 "+" 기호로 변환하지만, 나머지 문자는 모두 인코딩되지 않음을 명시함.

## herf 란?
태그의 href 속성은 링크된 페이지의 URL을 명시합니다.


