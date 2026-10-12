BTC xin tổng hợp nhanh một số nội dung chính của buổi mentoring:

1. Tổng quan bài toán và tiêu chí đánh giá

Làm rõ yêu cầu truy hồi tài liệu và đoạn nội dung y khoa từ truy vấn tiếng Việt trên kho dữ liệu Việt – Anh – Trung.
Giới thiệu cách đánh giá Document Retrieval và Chunk Retrieval thông qua Precision, Recall và F2-score.
Làm rõ nguyên tắc đối sánh nội dung và mối liên hệ giữa chunk với tài liệu gốc.
2. Thu thập và tiền xử lý dữ liệu

Giới thiệu các phương pháp thu thập, trích xuất nội dung, chuẩn hóa metadata và xử lý dữ liệu đa ngôn ngữ.
Chia sẻ cách chunking, loại bỏ nội dung nhiễu, xử lý dữ liệu trùng lặp và bảo toàn ngữ cảnh, thông tin gốc.
3. Định hướng xây dựng Baseline Retrieval

Phân nhóm tài liệu y khoa theo chủ đề, tham khảo hệ thống phân loại ICD-10.
Giới thiệu các phương pháp Lexical Retrieval (TF-IDF, BM25), Dense Retrieval (mBERT, BGE-M3).
Tham khảo hướng kết hợp Hybrid Retrieval và Reranking để cải thiện khả năng tìm kiếm xuyên ngôn ngữ và độ chính xác của kết quả.
Định hướng xây dựng pipeline Retrieval từ xử lý dữ liệu, lập chỉ mục đến truy hồi và xếp hạng kết quả.
4. Cập nhật nguồn dữ liệu cuộc thi

Nhằm giảm khối lượng dữ liệu cần thu thập, xử lý, đồng thời bổ sung nguồn tài liệu tiếng Anh cho kho dữ liệu truy hồi, Ban Tổ chức đã điều chỉnh danh sách nguồn dữ liệu của cuộc thi.

Các đội vui lòng sử dụng phiên bản mới của tệp links_corpus_vi_en_zh.parquet tại:

AIGuruTinix/ViBioMIR · Datasets at Hugging Face

Danh sách mới gồm 1.194.469 URL, với cơ cấu như sau:

Tiếng Việt: Giữ nguyên 714.469 URL.
Tiếng Trung: Rút gọn còn 240.000 URL.
Tiếng Anh: Bổ sung 240.000 URL.
Lưu ý: Mã định danh (ID) của các liên kết tài liệu không thay đổi. Các đội đã thu thập dữ liệu chỉ cần loại bỏ những tài liệu tiếng Trung đã bị rút khỏi danh sách và bổ sung tài liệu tiếng Anh vào kho dữ liệu truy hồi.

# **Một truy vấn, nhiều ngôn ngữ** 

▶ Đầu vào: một **truy vấn y khoa bằng tiếng Việt** . 

- Kho dữliệu: tài liệu từcác nguồn **Việt – Anh – Trung** . 

- Hệthống cần tìm đúng **tài liệu** và định vịđúng **đoạn nội dung gốc** . 

## Ví dụ 

Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận? 

# **Đánh giá ởhai cấp độ** 

## <mark>Document retrieval</mark> 

_D_ : tài liệu dựđoán. _G_ : tài liệu tham chiếu. 



<!-- Start of picture text -->
P = |D |  ∩ D| G | , R = |D |  ∩ G | G | .<br><!-- End of picture text -->

## <mark>Chunk retrieval</mark> 

Precision: tỷlệchunk nộp được tính đúng. Recall: tỷlệchunk tham chiếu tìm thấy. 



<!-- Start of picture text -->
Điểm F 2<br><!-- End of picture text -->



<!-- Start of picture text -->
5 P ( q ) R ( q )<br>F 2( q ) =<br>4 P ( q ) +  R ( q )<br>1<br>F 2 macro = � F 2( q )<br>|Q|<br>q∈Q<br>Tính từng truy vấn rồi lấy trung bình,<br>riêng cho tài liệu và đoạn.<br><!-- End of picture text -->


# **Ví dụ: đánh giá document** 

**Kịch bản minh họa:** _G_ = _{d_ 2 _, d_ 5 _, d_ 9 _}_ . 

Trả2 tài liệu _DA_ = _{d_ 2 _, d_ 5 _} P_ = 1, _R_ = 2 _/_ 3. _F_ 2 _≈_ 0 _,_ 714. 

Trả5 tài liệu _DB_ = _{d_ 2 _, d_ 5 _, d_ 9 _, d_ 11 _, d_ 12 _} P_ = 0 _,_ 6, _R_ = 1. _F_ 2 _≈_ 0 _,_ 882. 

Bài học 

Ví dụB tìm đủtài liệu liên quan nên có điểm cao hơn. Trảthêm tài liệu không liên quan vẫn làm precision giảm. 

# **Một chunk đúng phải khớp thếnào?** 

- Đối chiếu trong **cùng truy vấn và cùng doc_id** . 

- Chuẩn hóa NFKC, HTML entity, lowercase và khoảng trắng. 

<!-- Start of picture text -->
▶ LCS giữthứtựtoken nhưng không bắt buộc liên tiếp.<br>overlap( c, g ) = LCS tokens ( c, g ) ≥ 0 , 40<br>|g |<br><!-- End of picture text -->

Ví dụ 

Đoạn tham chiếu 100 token, LCS 45 token: overlap = 45%, đạt ngưỡng. 

# **Tổng quan dữliệu ViBioMIR** 

Kho tài liệu 

Tập truy vấn 

- **4.394.718 URL** từ **97 host** . 

- Mỗi tài liệu có một **ID** và một **URL** . 

- Nội dung gồm nhiều nguồn và loại tài liệu khác nhau. 

- **1.200 câu hỏi tiếng Việt** . 

- Độdài trung vị: **18 từ** . 

- Có câu hỏi ngắn và câu hỏi chứa nhiều ý. 

Điểm cần phân biệt 

Danh sách URL là điểm bắt đầu. Cần thu thập và xửlý nội dung trước khi có thểtìm kiếm. 

# **Dữliệu tập trung vào một sốnguồn lớn** 

<!-- Start of picture text -->
Nguồn SốURL<br>www.cnkang.com 963.438<br>www.120ask.com 918.479<br>www.familydoctor.com.cn 447.453<br>ask.39.net 337.375<br>zysjonline.com 243.347<br>Ý nghĩa<br>Hai host lớn nhất chiếm khoảng 42,8% sốURL. Cần hiểu cấu trúc nguồn lớn và vẫn<br>quan sát các nguồn nhỏđểtránh bỏsót loại tài liệu.<br><!-- End of picture text -->

# **Các loại nội dung trong corpus** 

- **Hỏi đáp y khoa** : câu hỏi, câu trảlời và thông tin người trảlời. 

- **Bài kiến thức** : bệnh, triệu chứng, chẩn đoán và điều trị. 

- **Sách và cẩm nang** : chương, mục, bảng và tài liệu tham khảo. 

- **Thuốc, dược liệu và tin tức** : cấu trúc và mục đích sửdụng khác nhau. 

Vì sao cần phân biệt? 

Mỗi loại tài liệu cần một cách trích xuất và chia đoạn phù hợp. 

# **Nội dung lấy từwebsite có dùng được ngay không?** 

- Một trang web gồm bài viết, menu, quảng cáo và nội dung liên quan. 

- Có trang chỉtrảvềthông báo hoặc màn hình chọn ngôn ngữ. 

- Tiêu đề, tác giảvà ngày đăng có thểthiếu hoặc không đúng với bài. 

- Cần nhận diện **phần nội dung chính** trước khi lập chỉmục. 

Câu hỏi cần kiểm tra 

Văn bản đang thu được có phải là bài viết cần tìm và còn giữđầy đủcấu trúc quan trọng không? 

**Ví dụ1: tải được trang nhưng chưa có bài viết** 

Nguyên nhân, triệu chứng và cách điều trịchóng mặt `https://www.vinmec.com/vie/bai-viet/chong-mat-vi` 

Bản thu thập ngày 11/10/2026 

Nếu đưa ngay vào chỉmục 

- Tiêu đềchỉcòn tên website. 

   - Không có kiến thức vềchóng mặt. 

- Trang thông báo chưa có nội dung tiếng Anh. 

   - Kết quảtìm kiếm có thểtrảvềthông báo thay cho bằng chứng. 

- Yêu cầu tiếp tục đểđọc tiếng Việt. 

# **Ví dụ2: chọn đúng vùng bài và giữcấu trúc** 

Bài viết trên Nhà thuốc Long Châu 

- 1 bánh trung thu bằng mấy bát cơm? Góc nhìn dinh dưỡng và sức khỏe. 

Những mục trong bài 

## Cách xửlý phù hợp 

- Thành phần dinh dưỡng trong 1 bánh Trung thu. 

- Bánh nướng và bánh dẻo. 

- Lượng calo trung bình. 

   - Lấy vùng bài viết thay vì toàn bộtrang. 

   - Giữtiêu đềmục cùng đoạn bên dưới. 

   - ▶ Chia đoạn theo từng mục đểtìm đúng nội dung. 

- Các câu hỏi thường gặp. 

# **Các cách thu thập dữliệu** 

- **Đọc HTML trực tiếp** : nhanh và đơn giản với trang tĩnh. Khó xửlý nội dung được tải bằng JavaScript. 

- **API công khai** : dữliệu thường có cấu trúc rõ. Không phải nguồn nào cũng cung cấp API. 

- **Trình duyệt tựđộng** : lấy được trang cần render. Tốn thời gian và tài nguyên hơn. 

- **Sitemap** : hỗtrợtìm danh sách trang. Thường không cung cấp đầy đủnội dung bài viết. 

Cách lựa chọn 

Khảo sát từng nguồn đểchọn cách thu thập phù hợp trước khi mởrộng quy mô. 

# **TừURL đến dữliệu có thểtìm kiếm** 

Danh sách URL _↓_ Thu thập nội dung _↓_ Trích xuất metadata và văn bản _↓_ Làm sạch và chuẩn hóa _↓_ Chia đoạn nội dung _↓_ **Lập chỉmục và truy hồi** 

# **Chuẩn hóa metadata** 

- **Thông tin mô tả** : tiêu đề, tóm tắt, chuyên mục và ngôn ngữ. 

- **Thông tin nguồn** : URL, tác giả, ngày đăng và ngày cập nhật. 

- **Thông tin tổchức** : loại tài liệu và nhóm chủđề. 

- Đưa các cách ghi khác nhau vềmột cấu trúc thống nhất. 

Tại sao cần metadata? 

Metadata giúp hiểu tài liệu, phân nhóm nội dung và hỗtrợtìm kiếm trước khi đọc toàn văn. 

# **Ví dụ: chuẩn hóa metadata** 

Cùng chủđề, giữnguyên tiêu đềvà ngôn ngữ VI Sỏi thận: triệu chứng và chẩn đoán EN Kidney stones: symptoms and diagnosis ZH 肾结石：症状与诊断 

Ngày đăng khác cách viết, cùng một định dạng 

15/09/2026 | September 15, 2026 | 2026 年 9 月 15 日 _−→_ `2026-09-15` 

**Giữriêng:** ID, URL, tiêu đềgốc, ngôn ngữvà ngày đăng. Thiếu thông tin thì đểtrống, không tựsuy tác giảhay ngày đăng. 

# **Chuẩn hóa nội dung văn bản** 

Bỏphần giao diện, giữphần bài viết Loại khỏi nội dung chính Giữtiêu đềbài VI  chủ/ Tìm kiếm Sỏi thận EN Home / Search Kidney stones ZH 首页 / 搜索 / 相关文章 肾结石 

Làm sạch cách biểu diễn 

`<p>Sỏi thận</p>` _→_ Sỏi thận 

`<p>Kidney stones</p>` _→_ Kidney stones <p> 肾结石 </p> _→_ 肾结石 

**Giữcấu trúc:** tiêu đềmục, đoạn văn, danh sách và bảng. 

# **Làm sạch nhưng không làm đổi nghĩa** 

Ba câu cùng mang ý phủđịnh 

VI **Không** ghi nhận sốt. EN **No** fever reported. 

ZH 未报告发热 

- Bỏtừphủđịnh sẽlàm thay đổi thông tin của câu. 

- Không xóa dấu tiếng Việt hoặc tựbỏcác từđược coi là ít quan trọng. 

- Giữnguyên sốvà đơn vị: **5 mg** khác **50 mg** . 

## Nguyên tắc 

Làm sạch định dạng, không viết lại ý nghĩa. Lưu văn bản gốc đểđối chiếu. 

# **Tiền xửlý dữliệu đa ngôn ngữ** 

Minh họa tách từcho tìm kiếm từkhóa Văn bản Một cách biểu diễn VI triệu chứng sỏi thận triệu_chứng | sỏi_thận EN kidney stones symptoms kidney | stones | symptoms ZH 肾结石症状 肾结石 | 症状 

- Tiếng Việt: khoảng trắng có thểnằm giữa các tiếng của một từ. 

- Tiếng Trung: nhiều từkhông được ngăn cách bằng khoảng trắng. 

- Dense retrieval dùng tokenizer đi kèm mô hình. 

**Cùng khái niệm:** sỏi thận _↔_ kidney stones _↔_ 肾结石 . 

**Chunking: chia tài liệu thành các đoạn** <mark>SS</mark> Các cách chia phổbiến Đánh 

Đánh đổi cần cân nhắc 

   - Đoạn quá ngắn dễmất ngữcảnh. 

- **Theo độdài** : đơn giản và dễkiểm soát. 

   - Đoạn quá dài dễchứa nhiều nội dung không liên quan. 

- **Theo cấu trúc** : dựa trên câu, đoạn và tiêu đềmục. 

   - Overlap giữngữcảnh nhưng tạo nội dung trùng. 

- **Theo ngữnghĩa** : gom các câu cùng chủđề. 

Tại sao cần chunking? 

Truy hồi đoạn giúp định vịbằng chứng cụthểthay vì trảvềcảtài liệu dài. 

# **Ví dụ: chia theo mục, giữngữcảnh** 



<!-- Start of picture text -->
Một bài, hai mục ởcảba ngôn ngữ<br>Tiêu đềbài Mục 1 Mục 2<br>VI Sỏi thận Triệu chứng Chẩn đoán<br>EN Kidney stones Symptoms Diagnosis<br>ZH 肾结石 症状 诊断<br><!-- End of picture text -->

- **Chunk A** : mục Triệu chứng và các câu thuộc mục này. 

▶ **Chunk B** : mục Chẩn đoán và các câu thuộc mục này. **Câu hỏi vềtriệu chứng** nên tìm đến chunk A. GiữID và vịtrí gốc đểtruy ngược mỗi chunk vềtài liệu. 

# **Ví dụ: giữcâu hỏi cùng câu trảlời** 

Tiếng Việt 

**Hỏi:** Sỏi thận là gì? **Đáp:** Sỏi thận là các khối rắn hình thành trong thận. Tiếng Anh 

**Q:** What are kidney stones? **A:** Kidney stones are solid deposits that form in the kidneys. Tiếng Trung 

什么是肾结石？ 肾结石是在肾脏内形成的固体沉积物。 

**Cách chia:** giữcặp hỏi đáp cùng đoạn nếu độdài cho phép. 

# **Hậu xửlý và kiểm tra chất lượng dữliệu** 

Ví dụ: hai URL chứa cùng một bài tiếng Việt 

A: Sỏi thận là các khối rắn hình thành trong thận. B: Sỏi thận là các khối rắn hình thành trong thận. 

- _→_ Nhận diện bản trùng, vẫn giữánh xạvềcác ID và URL gốc. 

Cùng chủđềchưa chắc là bản trùng 

Sỏi thận | Kidney stones | 肾结石 

- _→_ Không xóa bài chỉvì tiêu đềgần nghĩa hoặc khác ngôn ngữ. 

- Kiểm tra trang lỗi, nội dung rỗng và đoạn mất ngữcảnh. 

- Giữnguồn của từng chunk đểtrảvềđúng bằng chứng. 

# **Vì sao dữliệu quyết định chất lượng truy hồi?** 

▶ Không thu thập được bài liên quan thì mô hình không thểtìm thấy bài đó. 

▶ Trích xuất thiếu đoạn quan trọng thì bằng chứng bịmất. 

▶ Nội dung nhiễu hoặc trùng làm kết quảtìm kiếm khó chính xác. 

- Chia đoạn không phù hợp làm mất ngữcảnh của câu trảlời. 

Thông điệp 

Mô hình tốt cần đi cùng một kho dữliệu có độphủvà chất lượng phù hợp. 

<!-- Start of picture text -->
Xây dựng baseline<br><!-- End of picture text -->

# **Phân nhóm chủđềdựa trên ICD-10** 

- ICD-10 tổchức bệnh và vấn đềsức khỏe theo **hệphân cấp** . 

- Có thểbắt đầu bằng nhóm lớn như **tuần hoàn, hô hấp, nội tiết và tiêu hóa** . 

- Dựa trên **tiêu đề, tóm tắt và chuyên mục** đểgán nhóm chủđề. 

- Một tài liệu có thểthuộc nhiều nhóm. 

Tại sao dùng ICD-10? 

Có một hệquy chiếu thống nhất đểtổchức dữliệu và phân tích kết quảgiữa các ngôn ngữ. 


# **Các cách phân nhóm tài liệu** 

Dựa trên từđiển và quy tắc 

## Dựa trên mô hình ngôn ngữ 

- Khớp tên bệnh và từđồng nghĩa. 

- Dễgiải thích và không cần nhiều dữ liệu gán nhãn. 

- Dễbỏsót cách diễn đạt mới hoặc từ nhập nhằng. 

- So khớp ngữnghĩa hoặc phân loại văn bản. 

- Nhận diện được nhiều cách diễn đạt. 

- Cần kiểm tra nhãn sai và khảnăng phù hợp y sinh. 

## Hướng tiếp cận 

Bắt đầu bằng nhóm lớn, kết hợp nhiều tín hiệu và giữtrạng thái chưa xác định khi thiếu bằng chứng. 

# **Ví dụ: nhóm chủđềICD-10 đa ngôn ngữ** 

Ba tiêu đềminh họa cùng một chủđề 

VI Đái tháo đường type 2 EN Type 2 diabetes 

ZH 2 型糖尿病 

- _−→_ Nhóm **E00–E90** : bệnh nội tiết, dinh dưỡng và chuyển hóa. 

- Từđiển đa ngôn ngữgiúp nối các tên gọi vềcùng nhóm. 

- Đọc thêm tóm tắt hoặc nội dung nếu tiêu đềchưa đủrõ. 

- Tài liệu có nhiều chủđềcó thểnhận nhiều nhóm. 

**Không suy chẩn đoán:** tiêu đề“đau ngực” chưa đủđểkết luận bệnh mạch vành. 

# **Dùng nhóm ICD-10 đểlàm gì?** 

- **Khám phá dữliệu** : nhóm bệnh nào nhiều hoặc ít tài liệu? 

- **Phân tích kết quả** : hệthống làm tốt và chưa tốt ởnhóm nào? 

- **Hỗtrợtìm kiếm** : ưu tiên tài liệu thuộc nhóm chủđềliên quan. 

- **Hỗtrợdữliệu đa ngôn ngữ** : kết nối các cách gọi vềcùng một nhóm. 

## Tránh lọc quá chặt 

Câu hỏi có thểliên quan nhiều bệnh. Phân nhóm sai không nên làm mất toàn bộtài liệu hữu ích. 

# **Các nhóm phương pháp trong baseline** 

Nhóm phương pháp Ý tưởng Ví dụ Lexical retrieval Khớp từkhóa TF-IDF, BM25 Dense retrieval So khớp vector ngữnghĩa mBERT, BGE-M3 Hybrid retrieval Kết hợp lexical và dense Hợp thứhạng, RRF Reranking Đọc kỹtập ứng viên Cross-encoder 

Mục đích của baseline 

Bắt đầu bằng phương pháp đơn giản đểhiểu dữliệu, sau đó bổsung thành phần khi giải quyết được một hạn chếcụthể. 

# **Kiến trúc tổng thểhệthống truy hồi** 

**Câu hỏi tiếng Việt** _↓_ Tìm theo từkhóa Tìm theo ngữnghĩa Lexical retrieval Dense retrieval _↓_ Hợp các tài liệu và đoạn ứng viên _↓_ Xếp hạng lại theo mức độliên quan _↓_ **Trảvềtài liệu và đoạn bằng chứng gốc** 

# **Lexical retrieval: TF-IDF và BM25** 

## <mark>TF-IDF</mark> 

## <mark>BM25</mark> 

- Ưu tiên từxuất hiện nhiều trong bài nhưng hiếm trong corpus. 

- Dễhiểu và dễtạo mốc so sánh. 

- Chưa mô hình hóa tốt ảnh hưởng của độdài và lặp từ. 

- Bổsung bão hòa tần suất từvà điều chỉnh độdài. 

- Phù hợp với tài liệu có độdài khác nhau. 

- Vẫn phụthuộc từkhóa xuất hiện trong văn bản. 

# **Lexical retrieval: điểm mạnh và điểm yếu** 

Điểm mạnh 

Điểm yếu 

- Nhanh và tương đối ít tốn tài nguyên. 

- Dễgiải thích vì sao tài liệu được chọn. 

- Mạnh với thuật ngữhoặc thực thể chính xác. 

- Khó khớp từđồng nghĩa và cách diễn đạt khác. 

- Hạn chếkhi query và tài liệu khác ngôn ngữ. 

- Không hiểu đầy đủngữcảnh của câu. 

Tại sao vẫn dùng? 

Là mốc so sánh đơn giản và bổsung khảnăng khớp chính xác mà tìm kiếm ngữnghĩa có thể bỏsót. 

# **Dense retrieval: tìm kiếm theo ngữnghĩa** 

- Mã hóa câu hỏi và đoạn văn thành **vector** . 

- Các cách diễn đạt gần nghĩa được kỳvọng nằm gần nhau trong không gian vector. 

- Tìm các đoạn có vector gần với vector của câu hỏi. 

- Ví dụmô hình: **mBERT** và **BGE-M3** . 

## Ví dụ 

Sỏi thận _↔_ kidney stones _↔_ 肾结石 . Ba cách viết có thểbiểu đạt cùng một khái niệm. 

# **Dense retrieval: điểm mạnh và điểm yếu** 

Điểm mạnh 

Điểm yếu 

- Khớp cách diễn đạt khác nhau. 

- Có thểhỗtrợtruy hồi xuyên ngôn ngữ. 

- Tốn tài nguyên mã hóa và lưu vector. 

- ▶ Dễnhầm đoạn gần nghĩa nhưng sai chi tiết. 

- Phụthuộc mô hình và miền dữliệu. 

Tại sao dùng? 

Giảm khoảng cách ngôn ngữgiữa query và tài liệu. mBERT cần cách tạo vector và huấn luyện phù hợp. BGE-M3 được huấn luyện cho retrieval. 

# **Hybrid retrieval: kết hợp hai cách tìm kiếm** 

Ý tưởng và điểm mạnh 

Điểm yếu 

- Lấy ứng viên từlexical và dense. 

- Kết hợp khớp từchính xác với khớp ngữnghĩa. 

- Giảm phụthuộc vào một cách biểu diễn. 

- Phải vận hành nhiều chỉmục. 

- Cần chọn cách hợp kết quảphù hợp. 

- Không bảo đảm tốt hơn nếu ứng viên có nhiều nhiễu. 

Tại sao dùng? <mark>——K_—Vx<x</mark> Câu hỏi y khoa vừa cần đúng thuật ngữ, vừa có nhiều cách diễn đạt. <mark>ooo</mark> Có thểhợp thứhạng bằng RRF hoặc kết hợp điểm sau chuẩn hóa. 

# **Reranking: xếp hạng lại tập ứng viên** 

Ý tưởng và điểm mạnh 

Điểm yếu 

- Đọc đồng thời câu hỏi và từng đoạn ứng viên. 

- So khớp ngữcảnh chi tiết hơn. 

- Ưu tiên đoạn thực sựtrảlời đúng câu hỏi. 

- Chậm và tốn tính toán hơn retrieval. 

- Chỉxửlý một tập ứng viên nhỏ. 

- Không tìm lại được bằng chứng đã bị bỏsót. 

Tại sao dùng? 

Retrieval tìm rộng, reranking chọn kỹ. Ví dụmô hình đa ngôn ngữ: BGE-reranker-v2-m3. 

# **Ví dụ: cùng bệnh nhưng khác ý cần tìm** 

Câu hỏi tiếng Việt Triệu chứng của sỏi thận là gì? Ứng viên Chủđềcủa đoạn Mức liên quan kỳvọng A (VI) Chẩn đoán sỏi thận Cùng bệnh, khác ý B (EN) Symptoms of kidney stones Đúng ý triệu chứng C (ZH) 肾结石 : 症状 Đúng ý triệu chứng ▶ Lexical dễkhớp từtiếng Việt ởA nhưng có thểbỏsót B, C. ▶ Dense đa ngôn ngữcó thểđưa cảba đoạn vào tập ứng viên. 

- Reranker cần ưu tiên nội dung trảlời đúng ý, dù khác ngôn ngữ. 

# **Từkết quảtìm kiếm đến bằng chứng** 

▶ Gộp các đoạn thuộc cùng một tài liệu. 

- Loại các đoạn bịlặp do chia đoạn chồng lấn. 

▶ Chọn các đoạn liên quan đến từng ý trong câu hỏi. 

- Trảvềnội dung gốc và tài liệu nguồn tương ứng. 

Mục tiêu 

Tìm đúng tài liệu và đúng phần nội dung chứa thông tin cần tìm. 

# **Đánh giá các phương pháp như thếnào?** 

- So sánh trên **cùng corpus, cùng bộcâu hỏi và cùng cách chia đoạn** . 

- Đo **Precision, Recall và F** 2 ởcấp tài liệu và đoạn. 

- Xem thêm **thời gian tìm kiếm và tài nguyên sửdụng** . 

- Phân tích theo ngôn ngữ, nhóm bệnh và loại câu hỏi. 

Câu hỏi cần trảlời 

Phương pháp mới cải thiện ởđâu, còn sai ởđâu và có đáng với chi phí bổsung không? 

# **Ba điểm cần nhớ** 

- **Dữliệu** : thu thập đủvà giữđúng nội dung trước khi tìm kiếm. 

▶ **ICD-10** : giúp tổchức chủđềvà hiểu hệthống đang làm tốt ởnhóm nào. 

- **Phương pháp** : lexical khớp từ, dense khớp nghĩa, hybrid kết hợp và reranker chọn kỹ. 

Thông điệp 

Chọn phương pháp theo đặc điểm dữliệu và hạn chếcần giải quyết, sau đó kiểm chứng bằng thực nghiệm. 

# **ICD-10: các chương I–XI** 

- **I–IV:** I · A00–B99: nhiễm trùng / ký sinh. II · C00–D48: khối u. III · D50–D89: máu / miễn dịch. IV · E00–E90: nội tiết / chuyển hóa 

- **V–VIII:** V · F00–F99: tâm thần / hành vi. VI · G00–G99: thần kinh. VII · H00–H59: mắt. VIII · H60–H95: tai 

- **IX–XI:** IX · I00–I99: tuần hoàn. X · J00–J99: hô hấp. XI · K00–K93: tiêu hóa 

# **ICD-10: các chương XII–XXII** 

- **XII–XV:** XII · L00–L99: da. XIII · M00–M99: cơ xương / mô liên kết. XIV · N00–N99: tiết niệu / sinh dục. XV · O00–O99: thai sản 

- **XVI–XIX:** XVI · P00–P96: chu sinh. XVII · Q00–Q99: dịtật bẩm sinh. XVIII · R00–R99: triệu chứng / dấu hiệu. XIX · S00–T98: chấn thương / ngộđộc 

- **XX–XXII:** XX · V01–Y98: nguyên nhân bên ngoài. XXI · Z00–Z99: yếu tốsức khỏe / tiếp xúc dịch vụ. XXII · U00–U85: mã mục đích đặc biệt 

# **Phụlục: TF-IDF và BM25** 

## <mark>TF-IDF</mark> 

## <mark>BM25</mark> 

_wt,d_ = tf( _t, d_ ) idf( _t_ ) 

   - Sửdụng độhiếm của từ. 

   - Giảm lợi ích của việc lặp từquá nhiều. 

- TF: mức xuất hiện trong tài liệu. 

   - Điều chỉnh theo độdài tài liệu. 

- IDF: mức phân biệt trong corpus. 

- So sánh bằng cosine. 

Khác biệt chính 

BM25 bổsung bão hòa tần suất từvà chuẩn hóa độdài thay vì chỉdựa trên trọng sốTF-IDF. 

# **Tài liệu tham khảo** 

- **ViBioMIR** : README và báo cáo kiểm kê dữliệu. 

- **Cuộc thi R2AI2026** : mô tảnhiệm vụvà phương pháp đánh giá. 

- **WHO ICD-10** : hệphân loại bệnh và vấn đềsức khỏe. 

- **Introduction to Information Retrieval** : lexical retrieval. 

- **Model cards** : mBERT, BGE-M3 và BGE-reranker. 

- **Ví dụthuật ngữ** : NIDDK, MSD Manuals và WHO. 